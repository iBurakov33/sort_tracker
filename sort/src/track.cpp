#include "track.h"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {
cv::Rect ClampRectToFrame(const cv::Rect& bbox, const cv::Size& frame_size) {
    const cv::Rect frame_rect(0, 0, frame_size.width, frame_size.height);
    return bbox & frame_rect;
}

cv::Mat ComputeLbpHistogram(const cv::Mat& gray) {
    constexpr int kBins = 16;
    cv::Mat hist = cv::Mat::zeros(1, kBins, CV_32F);
    if (gray.rows < 3 || gray.cols < 3) {
        return hist;
    }

    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            const uchar center = gray.at<uchar>(y, x);
            unsigned char code = 0;
            code |= (gray.at<uchar>(y - 1, x - 1) >= center) << 7;
            code |= (gray.at<uchar>(y - 1, x) >= center) << 6;
            code |= (gray.at<uchar>(y - 1, x + 1) >= center) << 5;
            code |= (gray.at<uchar>(y, x + 1) >= center) << 4;
            code |= (gray.at<uchar>(y + 1, x + 1) >= center) << 3;
            code |= (gray.at<uchar>(y + 1, x) >= center) << 2;
            code |= (gray.at<uchar>(y + 1, x - 1) >= center) << 1;
            code |= (gray.at<uchar>(y, x - 1) >= center);
            hist.at<float>(0, code >> 4) += 1.0f;
        }
    }

    const float norm = static_cast<float>(cv::norm(hist, cv::NORM_L1));
    if (norm > 1e-6f) {
        hist /= norm;
    }
    return hist;
}
}  // namespace



Track::Track() : kf_(12, 4) {

    /*** Define constant acceleration model ***/
    // state - center_x, center_y, width, height,
    //         v_cx, v_cy, v_width, v_height,
    //         a_cx, a_cy, a_width, a_height
    constexpr float dt = 1.0f;
    constexpr float half_dt_sq = 0.5f * dt * dt;

    kf_.F_ = Eigen::MatrixXd::Identity(12, 12);
    for (int i = 0; i < 4; ++i) {
        kf_.F_(i, i + 4) = dt;
        kf_.F_(i, i + 8) = half_dt_sq;
        kf_.F_(i + 4, i + 8) = dt;
    }

    // Give high uncertainty to the unobservable initial velocities/accelerations
    kf_.P_ = Eigen::MatrixXd::Identity(12, 12);
    kf_.P_.diagonal().head(4).setConstant(10.0);
    kf_.P_.diagonal().segment(4, 4).setConstant(10000.0);
    kf_.P_.diagonal().tail(4).setConstant(10000.0);

    kf_.H_ = Eigen::MatrixXd::Zero(4, 12);
    kf_.H_(0, 0) = 1.0;
    kf_.H_(1, 1) = 1.0;
    kf_.H_(2, 2) = 1.0;
    kf_.H_(3, 3) = 1.0;

    kf_.Q_ = Eigen::MatrixXd::Identity(12, 12);
    kf_.Q_.diagonal().head(4).setConstant(1.0);
    kf_.Q_.diagonal().segment(4, 4).setConstant(0.05);
    kf_.Q_.diagonal().tail(4).setConstant(0.01);

    kf_.R_ <<
           1, 0, 0,  0,
            0, 1, 0,  0,
            0, 0, 10, 0,
            0, 0, 0,  10;
}


// Get predicted locations from existing trackers
// dt is encoded in F_ (currently fixed to 1.0)
void Track::Predict() {
    // When an object is temporarily lost (coasting), aggressively damp size dynamics
    // so predicted width/height change much less between frames.
    if (coast_cycles_ > 0) {
        constexpr double kSizeVelocityDamping = 0.2;
        constexpr double kSizeAccelerationDamping = 0.1;
        kf_.x_(6) *= kSizeVelocityDamping;   // v_width
        kf_.x_(7) *= kSizeVelocityDamping;   // v_height
        kf_.x_(10) *= kSizeAccelerationDamping; // a_width
        kf_.x_(11) *= kSizeAccelerationDamping; // a_height
    }

    kf_.Predict();

    // hit streak count will be reset
    if (coast_cycles_ > 0) {
        hit_streak_ = 0;
    }
    // accumulate coast cycle count
    coast_cycles_++;
}


// Update matched trackers with assigned detections
void Track::Update(const cv::Rect& bbox) {

    // get measurement update, reset coast cycle count
    coast_cycles_ = 0;
    // accumulate hit streak count
    hit_streak_++;

    // observation - center_x, center_y, width, height
    Eigen::VectorXd observation = ConvertBboxToObservation(bbox);
    kf_.Update(observation);

    if (has_observation_) {
        previous_observation_ = last_observation_;
    }
    last_observation_ = bbox;
    has_observation_ = true;

}

void Track::Update(const cv::Rect& bbox, const cv::Mat& frame) {
    Update(bbox);
    appearance_descriptor_ = ExtractAppearanceDescriptor(bbox, frame);
}


// Create and initialize new trackers for unmatched detections, with initial bounding box
void Track::Init(const cv::Rect &bbox) {
    kf_.x_.head(4) << ConvertBboxToObservation(bbox);
    hit_streak_++;
    previous_observation_ = bbox;
    last_observation_ = bbox;
    has_observation_ = true;
}

void Track::Init(const cv::Rect& bbox, const cv::Mat& frame) {
    Init(bbox);
    appearance_descriptor_ = ExtractAppearanceDescriptor(bbox, frame);
}


/**
 * Returns the current bounding box estimate
 * @return
 */
cv::Rect Track::GetStateAsBbox() const {
    return ConvertStateToBbox(kf_.x_);
}


cv::Rect Track::GetLastObservation() const {
    if (!has_observation_) {
        return GetStateAsBbox();
    }
    return last_observation_;
}


cv::Point2f Track::GetObservationDirection() const {
    if (!has_observation_) {
        return cv::Point2f(0.0f, 0.0f);
    }

    const cv::Point2f curr_center = GetCenter(last_observation_);
    const cv::Point2f prev_center = GetCenter(previous_observation_);
    const cv::Point2f motion = curr_center - prev_center;
    const float norm = std::sqrt(motion.x * motion.x + motion.y * motion.y);

    if (norm < 1e-6f) {
        return cv::Point2f(0.0f, 0.0f);
    }

    return cv::Point2f(motion.x / norm, motion.y / norm);
}


float Track::GetNIS() const {
    return kf_.NIS_;
}


float Track::CalculateAppearanceSimilarity(const cv::Rect& bbox, const cv::Mat& frame) const {
    if (appearance_descriptor_.empty()) {
        return 0.0f;
    }

    const cv::Mat candidate_descriptor = ExtractAppearanceDescriptor(bbox, frame);
    if (candidate_descriptor.empty()) {
        return 0.0f;
    }

    const float distance = static_cast<float>(cv::norm(appearance_descriptor_, candidate_descriptor, cv::NORM_L2));
    const float score = 1.0f - std::min(1.0f, distance / 2.0f);
    return std::max(0.0f, std::min(1.0f, score));
}


bool Track::HasAppearanceDescriptor() const {
    return !appearance_descriptor_.empty();
}


/**
 * Takes a bounding box in the form [x, y, width, height] and returns z in the form
 * [x, y, s, r] where x,y is the centre of the box and s is the scale/area and r is
 * the aspect ratio
 *
 * @param bbox
 * @return
 */
Eigen::VectorXd Track::ConvertBboxToObservation(const cv::Rect& bbox) const{
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(4);
    auto width = static_cast<float>(bbox.width);
    auto height = static_cast<float>(bbox.height);
    float center_x = bbox.x + width / 2;
    float center_y = bbox.y + height / 2;
    observation << center_x, center_y, width, height;
    return observation;
}


/**
 * Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
 * [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
 *
 * @param state
 * @return
 */
cv::Rect Track::ConvertStateToBbox(const Eigen::VectorXd &state) const {
    // state - center_x, center_y, width, height, v_*, a_*
    auto width = static_cast<int>(state[2]);
    auto height = static_cast<int>(state[3]);
    auto tl_x = static_cast<int>(state[0] - width / 2.0);
    auto tl_y = static_cast<int>(state[1] - height / 2.0);
    cv::Rect rect(cv::Point(tl_x, tl_y), cv::Size(width, height));
    return rect;
}


cv::Point2f Track::GetCenter(const cv::Rect& bbox) {
    return cv::Point2f(
            static_cast<float>(bbox.x) + static_cast<float>(bbox.width) * 0.5f,
            static_cast<float>(bbox.y) + static_cast<float>(bbox.height) * 0.5f);
}


cv::Mat Track::ExtractAppearanceDescriptor(const cv::Rect& bbox, const cv::Mat& frame) const {
    if (frame.empty()) {
        return cv::Mat();
    }

    const cv::Rect roi = ClampRectToFrame(bbox, frame.size());
    if (roi.width <= 2 || roi.height <= 2) {
        return cv::Mat();
    }

    constexpr int kNumStripes = 3;
    constexpr int kHBins = 8;
    constexpr int kSBins = 8;
    constexpr int kLbpBins = 16;
    constexpr int kDimsPerStripe = kHBins + kSBins + kLbpBins;
    cv::Mat descriptor = cv::Mat::zeros(1, kNumStripes * kDimsPerStripe, CV_32F);

    const cv::Mat patch = frame(roi);
    const int stripe_height = std::max(1, patch.rows / kNumStripes);
    for (int stripe = 0; stripe < kNumStripes; ++stripe) {
        const int y = stripe * stripe_height;
        const int h = (stripe == kNumStripes - 1) ? (patch.rows - y) : stripe_height;
        if (h <= 0) {
            continue;
        }

        const cv::Mat stripe_patch = patch(cv::Rect(0, y, patch.cols, h));

        cv::Mat hsv;
        cv::cvtColor(stripe_patch, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;
        cv::split(hsv, channels);

        cv::Mat h_hist;
        cv::Mat s_hist;
        const int h_bins[] = {kHBins};
        const int s_bins[] = {kSBins};
        const float h_range[] = {0.0f, 180.0f};
        const float s_range[] = {0.0f, 256.0f};
        const float* h_ranges[] = {h_range};
        const float* s_ranges[] = {s_range};
        const int h_channels[] = {0};
        const int s_channels[] = {0};
        cv::calcHist(&channels[0], 1, h_channels, cv::Mat(), h_hist, 1, h_bins, h_ranges, true, false);
        cv::calcHist(&channels[1], 1, s_channels, cv::Mat(), s_hist, 1, s_bins, s_ranges, true, false);
        const float h_norm = static_cast<float>(cv::norm(h_hist, cv::NORM_L1));
        const float s_norm = static_cast<float>(cv::norm(s_hist, cv::NORM_L1));
        if (h_norm > 1e-6f) {
            h_hist /= h_norm;
        }
        if (s_norm > 1e-6f) {
            s_hist /= s_norm;
        }

        cv::Mat gray;
        cv::cvtColor(stripe_patch, gray, cv::COLOR_BGR2GRAY);
        cv::Mat lbp_hist = ComputeLbpHistogram(gray);

        const int offset = stripe * kDimsPerStripe;
        h_hist.reshape(1, 1).copyTo(descriptor.colRange(offset, offset + kHBins));
        s_hist.reshape(1, 1).copyTo(descriptor.colRange(offset + kHBins, offset + kHBins + kSBins));
        lbp_hist.copyTo(descriptor.colRange(offset + kHBins + kSBins,
                                            offset + kHBins + kSBins + kLbpBins));
    }

    const float descriptor_norm = static_cast<float>(cv::norm(descriptor, cv::NORM_L2));
    if (descriptor_norm > 1e-6f) {
        descriptor /= descriptor_norm;
    }
    return descriptor;
}
