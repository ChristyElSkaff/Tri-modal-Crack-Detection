#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <algorithm>   // std::max

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

extern "C" {
  #include <seekcamera/seekcamera.h>
  #include <seekcamera/seekcamera_manager.h>
}

class SeekThermalNode : public rclcpp::Node {
public:
  SeekThermalNode()
  : Node("seek_thermal_cpp_node"),
    manager_(nullptr),
    camera_(nullptr),
    is_live_(false),
    frame_count_(0)
  {
    RCLCPP_INFO(this->get_logger(), "SeekThermal C++ node starting...");

    // Publisher for thermal image (temps in °C as float per pixel)
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/seek_thermal/image_raw", 10);

    // ---- Saving parameters ----
    // save_enable: enable writing frames to disk
    // save_dir: output directory
    // save_format: "exr" (recommended) or "csv" (huge)
    // save_every_n: save every Nth frame
    this->declare_parameter<bool>("save_enable", false);
    this->declare_parameter<std::string>("save_dir", "/tmp/seek_thermal");
    this->declare_parameter<std::string>("save_format", "exr");
    this->declare_parameter<int>("save_every_n", 1);

    // Create save directory if saving is enabled
    if (this->get_parameter("save_enable").as_bool()) {
      const auto save_dir = this->get_parameter("save_dir").as_string();
      std::error_code ec;
      std::filesystem::create_directories(save_dir, ec);
      if (ec) {
        RCLCPP_WARN(this->get_logger(),
                    "Could not create save_dir '%s': %s",
                    save_dir.c_str(), ec.message().c_str());
      } else {
        RCLCPP_INFO(this->get_logger(), "Saving enabled. Directory: %s", save_dir.c_str());
      }
    }

    // ---- Create camera manager (USB discovery) ----
    seekcamera_io_type_t discovery_mode =
        static_cast<seekcamera_io_type_t>(SEEKCAMERA_IO_TYPE_USB);

    seekcamera_error_t status =
        seekcamera_manager_create(&manager_, discovery_mode);

    if (status != SEEKCAMERA_SUCCESS || manager_ == nullptr) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to create Seek camera manager: %s",
                   seekcamera_error_get_str(status));
      throw std::runtime_error("seekcamera_manager_create failed");
    }

    // Register event callback (CONNECT / DISCONNECT / ERROR)
    status = seekcamera_manager_register_event_callback(
        manager_,
        &SeekThermalNode::camera_event_callback_trampoline,
        this);

    if (status != SEEKCAMERA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to register camera event callback: %s",
                   seekcamera_error_get_str(status));
      seekcamera_manager_destroy(&manager_);
      manager_ = nullptr;
      throw std::runtime_error("seekcamera_manager_register_event_callback failed");
    }

    RCLCPP_INFO(this->get_logger(),
                "Seek camera manager created (USB discovery). Waiting for camera connect...");
  }

  ~SeekThermalNode() override {
    RCLCPP_INFO(this->get_logger(), "Shutting down SeekThermal C++ node...");

    if (is_live_ && camera_ != nullptr) {
      seekcamera_capture_session_stop(camera_);
      is_live_ = false;
    }

    if (manager_ != nullptr) {
      seekcamera_error_t status = seekcamera_manager_destroy(&manager_);
      if (status != SEEKCAMERA_SUCCESS) {
        RCLCPP_WARN(this->get_logger(),
                    "Failed to destroy camera manager cleanly: %s",
                    seekcamera_error_get_str(status));
      }
      manager_ = nullptr;
    }
  }

private:
  // =====================
  // Manager event callback
  // =====================
  static void camera_event_callback_trampoline(
      seekcamera_t* camera,
      seekcamera_manager_event_t event,
      seekcamera_error_t event_status,
      void* user_data)
  {
    auto* self = static_cast<SeekThermalNode*>(user_data);
    if (self != nullptr) {
      self->handle_camera_event(camera, event, event_status);
    }
  }

  void handle_camera_event(
      seekcamera_t* camera,
      seekcamera_manager_event_t event,
      seekcamera_error_t event_status)
  {
    seekcamera_chipid_t cid;
    seekcamera_get_chipid(camera, &cid);

    RCLCPP_INFO(this->get_logger(),
                "Camera event: %s for camera %s",
                seekcamera_manager_get_event_str(event),
                cid);

    switch (event) {
      case SEEKCAMERA_MANAGER_EVENT_CONNECT:
        handle_camera_connect(camera);
        break;

      case SEEKCAMERA_MANAGER_EVENT_DISCONNECT:
        handle_camera_disconnect(camera);
        break;

      case SEEKCAMERA_MANAGER_EVENT_ERROR:
        RCLCPP_ERROR(this->get_logger(),
                     "Camera ERROR (%s): %s",
                     cid,
                     seekcamera_error_get_str(event_status));
        break;

      default:
        break;
    }
  }

  // =====================
  // CONNECT
  // =====================
  void handle_camera_connect(seekcamera_t* camera) {
    seekcamera_chipid_t cid;
    seekcamera_get_chipid(camera, &cid);

    RCLCPP_INFO(this->get_logger(), "Camera CONNECT: %s", cid);

    camera_ = camera;
    is_live_ = false;

    // Register frame callback
    seekcamera_error_t status = seekcamera_register_frame_available_callback(
        camera_,
        &SeekThermalNode::frame_available_callback_trampoline,
        this);

    if (status != SEEKCAMERA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to register frame callback for %s: %s",
                   cid,
                   seekcamera_error_get_str(status));
      return;
    }

    // Start capture session: THERMOGRAPHY_FLOAT
    const uint32_t frame_format = SEEKCAMERA_FRAME_FORMAT_THERMOGRAPHY_FLOAT;

    status = seekcamera_capture_session_start(camera_, frame_format);
    if (status != SEEKCAMERA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to start capture session for %s: %s",
                   cid,
                   seekcamera_error_get_str(status));
      is_live_ = false;
      return;
    }

    is_live_ = true;
    RCLCPP_INFO(this->get_logger(), "Capture session started for %s", cid);
  }

  // =====================
  // DISCONNECT
  // =====================
  void handle_camera_disconnect(seekcamera_t* camera) {
    seekcamera_chipid_t cid;
    seekcamera_get_chipid(camera, &cid);

    RCLCPP_INFO(this->get_logger(), "Camera DISCONNECT: %s", cid);

    if (camera_ == camera && is_live_) {
      seekcamera_capture_session_stop(camera_);
      is_live_ = false;
      camera_ = nullptr;
      RCLCPP_INFO(this->get_logger(), "Capture session stopped for %s", cid);
    }
  }

  // =====================
  // Frame callback
  // =====================
  static void frame_available_callback_trampoline(
      seekcamera_t* camera,
      seekcamera_frame_t* camera_frame,
      void* user_data)
  {
    auto* self = static_cast<SeekThermalNode*>(user_data);
    if (self != nullptr) {
      self->handle_frame(camera, camera_frame);
    }
  }

  static std::string stamp_to_string(const rclcpp::Time& t) {
    // Use nanoseconds integer for a safe unique filename
    std::ostringstream ss;
    ss << t.nanoseconds();
    return ss.str();
  }

  void save_frame_to_exr(const cv::Mat& temp_img, const std::string& path) {
    // EXR preserves float temps per pixel (best)
    // Requires OpenCV built with EXR support.
    if (!cv::imwrite(path, temp_img)) {
      RCLCPP_WARN(this->get_logger(), "Failed to write EXR: %s", path.c_str());
    }
  }

  void save_frame_to_csv(const cv::Mat& temp_img, const std::string& path) {
    // CSV is huge/slow but human-readable
    std::ofstream out(path);
    if (!out.is_open()) {
      RCLCPP_WARN(this->get_logger(), "Failed to open CSV for writing: %s", path.c_str());
      return;
    }

    out << std::fixed << std::setprecision(3);
    for (int y = 0; y < temp_img.rows; ++y) {
      const float* row = temp_img.ptr<float>(y);
      for (int x = 0; x < temp_img.cols; ++x) {
        out << row[x];
        if (x < temp_img.cols - 1) out << ",";
      }
      out << "\n";
    }
    out.close();
  }

  void handle_frame(seekcamera_t* camera, seekcamera_frame_t* camera_frame) {
    if (!is_live_ || camera != camera_) {
      return;
    }

    // Get the thermography float frame
    seekframe_t* frame = nullptr;
    seekcamera_error_t status = seekcamera_frame_get_frame_by_format(
        camera_frame,
        SEEKCAMERA_FRAME_FORMAT_THERMOGRAPHY_FLOAT,
        &frame);

    if (status != SEEKCAMERA_SUCCESS || frame == nullptr) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to get thermography frame: %s",
                   seekcamera_error_get_str(status));
      return;
    }

    const size_t width  = seekframe_get_width(frame);
    const size_t height = seekframe_get_height(frame);

    // Build CV_32FC1 (temp in °C as float per pixel)
    cv::Mat temp_img(static_cast<int>(height),
                     static_cast<int>(width),
                     CV_32FC1);

    for (size_t y = 0; y < height; ++y) {
      float* src_row = reinterpret_cast<float*>(seekframe_get_row(frame, y));
      float* dst_row = temp_img.ptr<float>(static_cast<int>(y));
      std::memcpy(dst_row, src_row, width * sizeof(float));
    }

    // Publish ROS Image (32FC1)
    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "seek_thermal_frame";

    auto msg = cv_bridge::CvImage(header, "32FC1", temp_img).toImageMsg();
    image_pub_->publish(*msg);

    // Save (optional)
    const bool save_enable = this->get_parameter("save_enable").as_bool();

    // FIX: make std::max use int64_t because ROS2 parameter ints are int64_t
    const int save_every = static_cast<int>(
        std::max<int64_t>(1, this->get_parameter("save_every_n").as_int())
    );

    if (save_enable) {
      frame_count_++;

      if ((frame_count_ % static_cast<uint64_t>(save_every)) == 0) {
        const auto save_dir = this->get_parameter("save_dir").as_string();
        const auto fmt      = this->get_parameter("save_format").as_string();

        const std::string ts = stamp_to_string(this->now());

        if (fmt == "csv") {
          const std::string filepath = save_dir + "/thermal_" + ts + ".csv";
          save_frame_to_csv(temp_img, filepath);
          RCLCPP_INFO(this->get_logger(), "Saved thermal frame: %s", filepath.c_str());
        } else {
          const std::string filepath = save_dir + "/thermal_" + ts + ".exr";
          save_frame_to_exr(temp_img, filepath);
          RCLCPP_INFO(this->get_logger(), "Saved thermal frame: %s", filepath.c_str());
        }
      }
    }

    RCLCPP_DEBUG(this->get_logger(),
                 "Published thermal frame (%zux%zu)", width, height);
  }

  // Members
  seekcamera_manager_t* manager_;
  seekcamera_t* camera_;
  bool is_live_;

  uint64_t frame_count_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SeekThermalNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

