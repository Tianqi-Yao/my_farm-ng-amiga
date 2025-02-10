"""Example of a camera service client."""
# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np
import os

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.stamp import get_stamp_by_semantics_and_clock_type
from farm_ng.core.stamp import StampSemantics

class VideoRecorder:
    def __init__(self, save_path: Path, max_file_size: int, fps: int = 30):
        """初始化视频存储类
        Args:
            save_path (Path): 视频保存路径（文件夹）
            max_file_size (int): 单个视频文件最大大小MB
            continuous (bool): 是否持续保存
            fps (int): 视频帧率
        """
        self.save_path = save_path
        self.max_file_size = max_file_size * 1024 * 1024  # 转换为字节
        self.continuous = continuous
        self.fps = fps
        self.frame_size = (640, 480)  # 默认分辨率（稍后会根据第一帧调整）
        self.video_writer = None
        self.file_index = 0
        self.current_file_path = None
        self.start_new_video()

    def start_new_video(self):
        """创建一个新的视频文件"""
        self.file_index += 1
        self.current_file_path = self.save_path / f"video_{self.file_index}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
        self.video_writer = cv2.VideoWriter(str(self.current_file_path), fourcc, self.fps, self.frame_size)

    def write_frame(self, frame: np.ndarray):
        """写入视频帧，并检查是否超过大小限制"""
        if self.video_writer is None:
            return

        # 写入视频帧
        self.video_writer.write(frame)

        # 检查文件大小
        if os.path.exists(self.current_file_path) and os.path.getsize(self.current_file_path) >= self.max_file_size:
            self.video_writer.release()
            self.start_new_video()

    def close(self):
        """关闭视频写入"""
        if self.video_writer:
            self.video_writer.release()

async def main(service_config_path: Path, save_path: Path, max_file_size: int):
    """运行相机服务客户端，并保存视频"""
    config: EventServiceConfig = proto_from_json_file(service_config_path, EventServiceConfig())
    recorder = VideoRecorder(save_path, max_file_size)

    # ------------------------------ async for  持续监听  事件客户端 EventClient(config) 连接到相机服务，并订阅第一个数据流 config.subscriptions[0]----------------------------- #
    async for event, message in EventClient(config).subscribe(config.subscriptions[0], decode=True):
        # Find the monotonic driver receive timestamp, or the first timestamp if not available.
        # ----------------------------------- 获取时间戳 ---------------------------------- #
        stamp = (
            get_stamp_by_semantics_and_clock_type(event, StampSemantics.DRIVER_RECEIVE, "monotonic") # 数据被相机驱动接收的时间
            or event.timestamps[0].stamp
        )

        # Print the timestamp and metadata
        print(f"Timestamp: {stamp}\n")
        print(f"Meta: {message.meta}")
        print("###################\n")

        # Cast image data bytes to numpy and decode
        image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)  # 得到numpy数组, 解码图像数据
        # 处理深度图像
        if event.uri.path == "/disparity":   # 如果是视差图像 
            image = cv2.applyColorMap(image * 3, cv2.COLORMAP_JET)  # 视差图像使用伪彩色, 放大像素值（image * 3），增强视觉效果。
        
        # 设置正确的帧大小
        if recorder.frame_size != (image.shape[1], image.shape[0]):
            recorder.frame_size = (image.shape[1], image.shape[0])
            recorder.start_new_video()  # 重新初始化写入器以匹配分辨率

        # Visualize the image
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        cv2.waitKey(1)

        # 录像
        recorder.write_frame(image)


    recorder.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python main.py", description="Amiga camera-stream recorder.")
    parser.add_argument("--service-config", type=Path, required=True, help="The camera config.")
    parser.add_argument("--save-path", type=Path, default="videos", help="The directory to save videos.")
    parser.add_argument("--max-file-size", type=int, default=100, help="Max file size per video (MB).")
    args = parser.parse_args()

    # 创建保存路径
    args.save_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(main(args.service_config, args.save_path, args.max_file_size, args.continuous))
