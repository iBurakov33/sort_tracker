#pragma once

constexpr int kNumColors = 32;

constexpr int kMaxCoastCycles = 5;

constexpr int kMinHits = 1;

constexpr float kAssociationIouThreshold = 0.1f;

// Observation-centric blend factor: 0 => pure IoU(SORT), 1 => pure direction consistency
constexpr float kVelocityDirectionWeight = 0.2f;

// Set threshold to 0 to accept all detections
constexpr float kMinConfidence = 0.45f;
