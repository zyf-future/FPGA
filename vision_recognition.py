#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read camera/PX4 pose, report one two-digit target, record every received frame.

No arming, flight mode, Offboard heartbeat or trajectory command is published
by this process. Run with flight_control.py, not the old standalone controller.
"""
from __future__ import annotations
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import uuid
import mission_common as core

np=None
LOG=logging.getLogger("vision_recognition")
CONTEXT="/digit_mission/context"
VISION_STATUS="/digit_mission/vision_status"
TARGET="/digit_mission/target"


def make_node(c,folder,console,shared,recorder):
    from rclpy.node import Node
    from rclpy.clock import Clock,ClockType
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,HistoryPolicy
    from sensor_msgs.msg import Image,CameraInfo
    from std_msgs.msg import String
    from geometry_msgs.msg import PointStamped
    from px4_msgs.msg import VehicleLocalPosition,VehicleAttitude

    def qos(depth=1):
        return QoSProfile(depth=depth,history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)

    class VisionNode(Node):
        def __init__(self):
            super().__init__("digit_mission_vision")
            self.boot=uuid.uuid4().hex
            self.mission_id=""
            self.context=None
            self.context_rx=-math.inf
            self.ground=None
            self.seq=0
            self.pos_stamp=self.q_stamp=-math.inf
            self.pos_reset=self.q_reset=None
            self.last_target=-1
            self.last_report=-math.inf
            self.status_pub=self.create_publisher(String,VISION_STATUS,qos())
            self.target_pub=self.create_publisher(String,TARGET,qos(3))
            self.point_pub=self.create_publisher(PointStamped,"/object_position_ned",10)
            self.create_subscription(String,CONTEXT,self.on_context,qos())
            self.create_subscription(Image,c.image_topic,self.on_image,qos(c.image_qos_depth))
            self.create_subscription(CameraInfo,c.camera_info_topic,self.on_calibration,qos())
            self.create_subscription(VehicleLocalPosition,
                core.px4_topic("/fmu/out/vehicle_local_position",VehicleLocalPosition,c.position_topic),
                self.on_pos,qos(10))
            self.create_subscription(VehicleAttitude,
                core.px4_topic("/fmu/out/vehicle_attitude",VehicleAttitude,c.attitude_topic),
                self.on_q,qos(10))
            self.steady=Clock(clock_type=ClockType.STEADY_TIME)
            self.create_timer(.10,self.tick,clock=self.steady)
            LOG.info("vision-only node started; no flight-command publishers")
            if not c.require_camera_info:
                LOG.warning("inherited camera intrinsics used until valid CameraInfo arrives")

        def ros_now(self):
            return self.get_clock().now().nanoseconds/1e9

        def fresh(self,stamp):
            return stamp>0 and -.05<=self.ros_now()-stamp<=c.state_timeout

        def reset_histories(self,reason):
            with shared.lock:
                shared.positions.clear()
                shared.attitudes.clear()
                shared.generation+=1
                shared.target=None
                shared.last_pipeline_ok=-math.inf
            LOG.warning("pose histories reset: %s",reason)

        def on_pos(self,msg):
            stamp=(int(getattr(msg,"timestamp_sample",0)) or int(msg.timestamp))/1e6
            if stamp<=self.pos_stamp:
                return
            self.pos_stamp=stamp
            reset=(int(msg.xy_reset_counter),int(msg.z_reset_counter),int(msg.heading_reset_counter))
            if self.pos_reset is not None and reset!=self.pos_reset:
                self.reset_histories("local position reset")
            self.pos_reset=reset
            p=np.array([msg.x,msg.y,msg.z],float)
            if not msg.xy_valid or not msg.z_valid or not np.isfinite(p).all():
                with shared.lock:
                    shared.positions.clear()
                    shared.last_pipeline_ok=-math.inf
                return
            if self.fresh(stamp):
                with shared.lock:
                    shared.positions.append((stamp,p))

        def on_q(self,msg):
            stamp=(int(getattr(msg,"timestamp_sample",0)) or int(msg.timestamp))/1e6
            if stamp<=self.q_stamp:
                return
            self.q_stamp=stamp
            reset=int(msg.quat_reset_counter)
            if self.q_reset is not None and reset!=self.q_reset:
                self.reset_histories("attitude reset")
            self.q_reset=reset
            try:
                q=core.normalized_quaternion(msg.q)
            except ValueError:
                with shared.lock:
                    shared.attitudes.clear()
                    shared.last_pipeline_ok=-math.inf
                return
            if self.fresh(stamp):
                with shared.lock:
                    shared.attitudes.append((stamp,q))

        def on_context(self,msg):
            try:
                d=json.loads(msg.data)
                if d["schema"]!=1 or not isinstance(d["mission_id"],str) or not d["mission_id"]:
                    return
                age=self.ros_now()-float(d["stamp"])
                if not math.isfinite(age) or not -.05<=age<=1.:
                    return
                ground=d["ground_z"]
                if ground is not None and not math.isfinite(float(ground)):
                    return
                number=d["locked_number"]
                if number is not None and (type(number) is not int or not 10<=number<=99):
                    return
                target=None if d["target_ned"] is None else np.array(d["target_ned"],float)
                if target is not None and (target.shape!=(3,) or not np.isfinite(target).all()):
                    return
                if self.mission_id!=d["mission_id"] or self.ground!=ground:
                    if console.printed:
                        shared.fail("mission changed after one number was already accepted; restart both processes for a new mission")
                        return
                    with shared.lock:
                        shared.generation+=1
                        shared.target=None
                    self.last_target=-1
                self.mission_id=d["mission_id"]
                self.ground=ground
                self.context=d
                self.context_rx=time.monotonic()
                with shared.lock:
                    if ground is not None:
                        c.ground_z=float(ground)
                    shared.active=bool(d["detect"] is True and ground is not None and not shared.fault)
                    shared.locked_number=number
                    shared.locked_point=target
                if number is not None:
                    console.emit(number)  # flight accepted this particular target
            except (KeyError,TypeError,ValueError):
                LOG.warning("invalid mission context")

        def on_calibration(self,msg):
            try:
                K=np.asarray(msg.k,float).reshape(3,3).copy()
                D=np.asarray(msg.d,float)
                if D.size==0: D=np.zeros(5)
                model=str(msg.distortion_model)
                if model not in ("","plumb_bob","rational_polynomial","equidistant"):
                    raise ValueError("unsupported distortion model")
                if model=="equidistant" and D.size!=4 or model!="equidistant" and D.size not in (4,5,8,12,14):
                    raise ValueError("invalid distortion coefficient count")
                bx,by=max(1,int(msg.binning_x)),max(1,int(msg.binning_y))
                w,h=int(msg.roi.width) or int(msg.width),int(msg.roi.height) or int(msg.height)
                K[0,2]-=msg.roi.x_offset
                K[1,2]-=msg.roi.y_offset
                K[0,:]/=bx
                K[1,:]/=by
                if w<=0 or h<=0 or not np.isfinite(K).all() or not np.isfinite(D).all() or K[0,0]<=0 or K[1,1]<=0:
                    raise ValueError("invalid calibration")
                with shared.lock:
                    shared.calibration=(K,D,w/bx,h/by,model)
            except Exception as exc:
                shared.notice("calibration","CameraInfo rejected: "+str(exc))

        def on_image(self,msg):
            self.seq+=1
            stamp=float(msg.header.stamp.sec)+msg.header.stamp.nanosec/1e9
            packet=core.FramePacket(self.seq,msg,stamp,self.ros_now(),time.monotonic())
            with shared.lock:
                shared.stats["frames_received"]+=1
            recorder.submit(packet)
            shared.put_latest(packet)

        def tick(self):
            mono=time.monotonic()
            ros=self.ros_now()
            with shared.lock:
                context_fresh=mono-self.context_rx<=1.0
                if not context_fresh:
                    shared.active=False
                ready=bool(context_fresh and self.ground is not None and shared.model_ready
                           and mono-shared.last_pipeline_ok<=1.2 and not shared.fault)
                target=shared.target
                status=dict(shared.stats)
                status.update(schema=1,boot_id=self.boot,mission_id=self.mission_id,stamp=ros,
                              ready=ready,fault=shared.fault,model_ready=shared.model_ready,
                              jpeg_backlog=recorder.queue.qsize())
            sm=String()
            sm.data=json.dumps(status,allow_nan=False,separators=(",",":"))
            self.status_pub.publish(sm)
            if (ready and target is not None and mono-target.captured<=c.vision_timeout
                    and self.context is not None and self.context["detect"] is True):
                # Repeat recent results for best-effort delivery. Capture time
                # and frame ID do not change on retries.
                data=dict(schema=1,mission_id=self.mission_id,boot_id=self.boot,
                          seq=target.frame_id,number=target.number,
                          captured_ros=ros-(mono-target.captured),
                          confidence=target.confidence,ned=target.point.tolist())
                tm=String()
                tm.data=json.dumps(data,allow_nan=False,separators=(",",":"))
                self.target_pub.publish(tm)
                point=PointStamped()
                stamp=data["captured_ros"]
                point.header.stamp.sec=int(stamp)
                point.header.stamp.nanosec=int((stamp-int(stamp))*1e9)
                point.header.frame_id="px4_local_ned"
                point.point.x,point.point.y,point.point.z=map(float,target.point)
                self.point_pub.publish(point)
            if mono-self.last_report>=2.:
                self.last_report=mono
                path=folder/"status.tmp"
                path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
                path.replace(folder/"status.json")

    return VisionNode()


def main(argv=None):
    global np
    console=node=executor=recorder=shared=vision=ros=None
    code=0
    try:
        console=core.QuietConsole()
        c,ros_args=core.make_parser().parse_known_args(argv)
        if ros_args and ros_args[0]!="--ros-args":
            raise ValueError("unknown arguments: "+" ".join(ros_args))
        core.validate_config(c)
        folder=Path(c.save_dir).expanduser().resolve()/(time.strftime("vision_%Y%m%d_%H%M%S_")+str(os.getpid()))
        folder.mkdir(parents=True,exist_ok=False)
        frames=folder/"frames"
        frames.mkdir()
        console.redirect(folder/"run.log")
        logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
        (folder/"config.json").write_text(json.dumps(vars(c),indent=2),encoding="utf-8")
        os.environ.setdefault("YOLO_VERBOSE","False")
        os.environ.setdefault("OMP_NUM_THREADS","1")
        import numpy as n
        import cv2
        np=core.np=n
        core.cv2=cv2
        cv2.setNumThreads(1)
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        ros=rclpy
        ros.init(args=ros_args)
        shared=core.Shared()
        recorder=core.JpegRecorder(c,frames,shared)
        node=make_node(c,folder,console,shared,recorder)
        vision=core.VisionWorker(c,shared)
        vision.start()
        executor=SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        LOG.info("vision stopped by operator; drain queued JPEGs")
    except BaseException:
        import traceback
        traceback.print_exc()
        code=1
    finally:
        if shared is not None:
            shared.stop.set()
            with shared.condition: shared.condition.notify_all()
        if executor is not None: executor.shutdown(timeout_sec=2.)
        if vision is not None: vision.thread.join(timeout=5.)
        if recorder is not None: recorder.close()
        if node is not None: node.destroy_node()
        if ros is not None and ros.ok(): ros.shutdown()
        if console is not None: console.close()
    return code


if __name__=="__main__":
    raise SystemExit(main())
