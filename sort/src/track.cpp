#include "track.h"

#include <cmath>


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


// Create and initialize new trackers for unmatched detections, with initial bounding box
void Track::Init(const cv::Rect &bbox) {
    kf_.x_.head(4) << ConvertBboxToObservation(bbox);
    hit_streak_++;
    previous_observation_ = bbox;
    last_observation_ = bbox;
    has_observation_ = true;
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
