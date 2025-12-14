from inference import VideoTemporalInference

# Initialize
engine = VideoTemporalInference(
    # model_path="./ckpts/Qwen3-VL-4B-Instruct",
    model_path="./outputs/sft/checkpoint-final",
)

# Predict
result = engine.predict(
    video_path="./data/videos/timerft_data/_0yiT0hhCCM_00:06:44:200_00:07:09:100.mp4",
    query="the paper bill has an image of a woman holding up a paper",
    video_start=404.2,
    video_end=429.1
)

print(f"Segment: {result['start_time']:.2f}s - {result['end_time']:.2f}s")