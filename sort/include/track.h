#pragma once

#include <opencv2/core.hpp>
#include "kalman_filter.h"

class Track {
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void Init(const cv::Rect& bbox);
    void Predict();
    void Update(const cv::Rect& bbox);
    void Init(const cv::Rect& bbox, const cv::Mat& frame);
    void Update(const cv::Rect& bbox, const cv::Mat& frame);
    cv::Rect GetStateAsBbox() const;
    cv::Rect GetLastObservation() const;
    cv::Point2f GetObservationDirection() const;
    float GetNIS() const;
    float CalculateAppearanceSimilarity(const cv::Rect& bbox, const cv::Mat& frame) const;
    bool HasAppearanceDescriptor() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

private:
    Eigen::VectorXd ConvertBboxToObservation(const cv::Rect& bbox) const;
    cv::Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;
    static cv::Point2f GetCenter(const cv::Rect& bbox);
    cv::Mat ExtractAppearanceDescriptor(const cv::Rect& bbox, const cv::Mat& frame) const;

    KalmanFilter kf_;
    cv::Rect last_observation_;
    cv::Rect previous_observation_;
    cv::Mat appearance_descriptor_;
    bool has_observation_ = false;
};
