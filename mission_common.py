#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared vision and landing helpers for the two-process PX4 digit mission.
Imported by flight_control.py and vision_recognition.py; never starts a node.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Optional

np: Any = None
cv2: Any = None
LOG = logging.getLogger("digit_landing")
DEFAULT_MODEL = "/home/robot/px4_digit_mission/resource/best.pt"


class QuietConsole:
    """Redirect even native-library output; preserve a separate result fd."""

    def __init__(self):
        self.result_fd = os.dup(1)
        self.printed = False
        self.lock = threading.Lock()
        self.bootstrap = Path(tempfile.gettempdir()) / (
            "digit_px4_bootstrap_%d.log" % os.getpid()
        )
        self.redirect(self.bootstrap)

    @staticmethod
    def redirect(path):
        sys.stdout.flush()
        sys.stderr.flush()
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        finally:
            os.close(fd)

    def emit(self, number):
        with self.lock:
            if self.printed:
                return
            if not 10 <= int(number) <= 99:
                raise ValueError("result must be in 10..99")
            self.printed = True
            try:
                os.write(self.result_fd, ("%02d\n" % number).encode("ascii"))
            except OSError:
                LOG.exception("result output is unavailable")

    def close(self):
        os.close(self.result_fd)


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--class-map", default="")
    p.add_argument("--image-topic", default="/camera/image")
    p.add_argument("--camera-info-topic", default="/camera/camera_info")
    p.add_argument("--save-dir", default="/home/robot/px4_digit_mission/pictures")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--save-workers", type=int, default=2)
    p.add_argument("--save-queue-frames", type=int, default=256)
    p.add_argument("--image-qos-depth", type=int, default=8)
    p.add_argument("--ground-z", type=float, default=0.0)
    p.add_argument("--fx", type=float, default=539.936)
    p.add_argument("--fy", type=float, default=539.936)
    p.add_argument("--cx", type=float, default=640.0)
    p.add_argument("--cy", type=float, default=480.0)
    p.add_argument("--calibration-width", type=int, default=1280)
    p.add_argument("--calibration-height", type=int, default=960)
    p.add_argument("--require-camera-info", action="store_true")
    p.add_argument("--camera-yaw-deg", type=float, default=0.0)
    p.add_argument("--camera-offset-m", default="0,0,0")
    p.add_argument("--panel-rotation", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--segment-width", type=int, default=640)
    p.add_argument("--max-panels", type=int, default=4)
    p.add_argument("--min-confidence", type=float, default=0.60)
    p.add_argument("--fast-confidence", type=float, default=0.85)
    p.add_argument("--confirm-window", type=float, default=1.2)
    p.add_argument("--cluster-diameter", type=float, default=0.50)
    p.add_argument("--target-gate", type=float, default=0.75)
    p.add_argument("--max-target-range", type=float, default=30.0)
    p.add_argument("--max-frame-age", type=float, default=0.70)
    p.add_argument("--pose-tolerance", type=float, default=0.08)
    p.add_argument("--state-timeout", type=float, default=0.40)
    p.add_argument("--vision-timeout", type=float, default=0.90)
    p.add_argument("--horizontal-speed", type=float, default=0.60)
    p.add_argument("--descent-speed", type=float, default=0.25)
    p.add_argument("--align-radius", type=float, default=0.20)
    p.add_argument("--align-speed", type=float, default=0.20)
    p.add_argument("--align-seconds", type=float, default=0.80)
    p.add_argument("--handover-height", type=float, default=1.0)
    p.add_argument("--source-system", type=int, default=245)
    p.add_argument("--source-component", type=int, default=191)
    for name in ("position", "attitude", "status", "land", "ack",
                 "command", "setpoint", "offboard"):
        p.add_argument("--%s-topic" % name, default="")
    return p


def validate_config(c):
    positive = ("fx", "fy", "calibration_width", "calibration_height",
                "imgsz", "segment_width", "max_panels", "save_workers",
                "save_queue_frames", "image_qos_depth", "confirm_window",
                "cluster_diameter", "target_gate", "max_target_range",
                "max_frame_age", "pose_tolerance", "state_timeout",
                "vision_timeout", "horizontal_speed", "descent_speed",
                "align_radius", "align_speed", "align_seconds",
                "handover_height")
    for key in positive:
        if not math.isfinite(getattr(c, key)) or getattr(c, key) <= 0:
            raise ValueError("invalid positive parameter: " + key)
    for key in ("ground_z", "cx", "cy", "camera_yaw_deg"):
        if not math.isfinite(getattr(c, key)):
            raise ValueError("nonfinite parameter: " + key)
    if not 0 < c.min_confidence <= c.fast_confidence <= 1:
        raise ValueError("invalid confidence thresholds")
    if not 1 <= c.jpeg_quality <= 100:
        raise ValueError("jpeg quality must be 1..100")
    if not 1 <= c.source_system <= 255 or not 1 <= c.source_component <= 255:
        raise ValueError("invalid MAVLink sender IDs")
    offset = [float(x) for x in c.camera_offset_m.split(",")]
    if len(offset) != 3 or not all(math.isfinite(x) for x in offset):
        raise ValueError("camera offset must be three finite FRD values")
    c.camera_offset = offset


def class_lut(names, override=""):
    if isinstance(names, dict):
        if set(names) != set(range(10)):
            raise ValueError("model must have exactly ten contiguous classes 0..9")
        names = [str(names[i]).strip() for i in range(10)]
    else:
        names = [] if names is None else [str(x).strip() for x in names]
    if len(names) != 10:
        raise ValueError("four-class or non-ten-class model is not supported")
    if override.strip():
        values = [int(x.strip()) for x in override.split(",")]
    else:
        if not all(re.fullmatch(r"[0-9]", x) for x in names):
            raise ValueError("ambiguous class names; supply --class-map explicitly")
        values = [int(x) for x in names]
    if len(values) != 10 or set(values) != set(range(10)):
        raise ValueError("class map must contain every digit 0..9 exactly once")
    return tuple(values)


def image_to_bgr(msg):
    enc = str(msg.encoding).lower()
    channels = {"mono8": 1, "8uc1": 1, "bgr8": 3, "rgb8": 3,
                "8uc3": 3, "bgra8": 4, "rgba8": 4, "8uc4": 4}.get(enc)
    if channels is None:
        raise ValueError("unsupported image encoding: " + enc)
    h, w = int(msg.height), int(msg.width)
    row = w * channels
    step = int(msg.step) or row
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if h <= 0 or w <= 0 or step < row or raw.size < h * step:
        raise ValueError("malformed image size/step/data")
    a = np.ascontiguousarray(raw[:h * step].reshape(h, step)[:, :row]
                             .reshape(h, w, channels))
    if channels == 1:
        return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    conversion = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
                  "bgra8": cv2.COLOR_BGRA2BGR, "8uc4": cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(a, conversion[enc]) if enc in conversion else a


def normalized_quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("invalid quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("zero quaternion")
    return q / norm


def rotation(q):
    w, x, y, z = normalized_quaternion(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_of(q):
    w, x, y, z = normalized_quaternion(q)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def slerp(a, b, t):
    a, b = normalized_quaternion(a), normalized_quaternion(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(a + t * (b-a))
    theta = math.acos(dot)
    return (math.sin((1-t)*theta)*a + math.sin(t*theta)*b) / math.sin(theta)


def interpolate(samples, stamp, tolerance, quaternion=False):
    """Timestamp-aligned samples; bounded nearest endpoint, never extrapolate."""
    if not samples:
        return None
    i = bisect.bisect_left([s[0] for s in samples], stamp)
    if i < len(samples) and abs(samples[i][0] - stamp) < 1e-9:
        return samples[i][1].copy()
    if i == 0 or i == len(samples):
        item = samples[0 if i == 0 else -1]
        return item[1].copy() if abs(item[0]-stamp) <= tolerance else None
    a, b = samples[i-1], samples[i]
    if stamp-a[0] > tolerance or b[0]-stamp > tolerance:
        return None
    t = (stamp-a[0]) / (b[0]-a[0])
    return slerp(a[1], b[1], t) if quaternion else a[1]+t*(b[1]-a[1])


def scaled_calibration(c, info, width, height):
    if info is None:
        if c.require_camera_info:
            return None
        K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], float)
        iw, ih, D, model = c.calibration_width, c.calibration_height, np.zeros(5), "plumb_bob"
    else:
        K, D, iw, ih, model = info
        K = K.copy()
    if iw <= 0 or ih <= 0 or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0:
        return None
    K[0, :] *= width / iw
    K[1, :] *= height / ih
    return K, D, model


def project_ground(pixel, position, q, calibration, c):
    K, D, model = calibration
    uv = np.array(pixel, dtype=float).reshape(1, 1, 2)
    if not np.isfinite(uv).all() or not np.isfinite(position).all():
        return None
    if model == "equidistant":
        xy = cv2.fisheye.undistortPoints(uv, K, D.reshape(-1,1))[0,0]
    elif model in ("plumb_bob", "rational_polynomial", ""):
        xy = cv2.undistortPoints(uv, K, D)[0,0]
    else:
        return None
    rwb = rotation(q)
    a = math.radians(c.camera_yaw_deg)
    rz = np.array([[math.cos(a),-math.sin(a),0],
                   [math.sin(a), math.cos(a),0],[0,0,1]])
    rbc = rz @ np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
    ray = rwb @ rbc @ np.array([xy[0], xy[1], 1.0])
    origin = position + rwb @ np.array(c.camera_offset)
    if not np.isfinite(ray).all() or ray[2] <= 1e-5:
        return None
    depth = (c.ground_z-origin[2])/ray[2]
    if depth <= 0.05:
        return None
    target = origin + depth*ray
    if not np.isfinite(target).all() or np.linalg.norm(target[:2]-position[:2]) > c.max_target_range:
        return None
    target[2] = c.ground_z
    return target


def ordered_quad(points):
    q = np.asarray(points, dtype=np.float32).reshape(4,2)
    center = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:,1]-center[1], q[:,0]-center[0]))]
    q = np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)
    return q


def red_white_panels(frame, c):
    """Inspect several red candidates; rectify white panels by homography."""
    h, w = frame.shape[:2]
    scale = min(1.0, c.segment_width / w)
    small = cv2.resize(frame, (round(w*scale), round(h*scale))) if scale < 1 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0,80,70), (10,255,255)) | cv2.inRange(hsv,(170,80,70),(180,255,255))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    panels = []
    for sc in contours:
        if cv2.contourArea(sc) < max(12,80*scale*scale):
            break
        contour = (sc.astype(float)/scale).round().astype(np.int32)
        x,y,rw,rh = cv2.boundingRect(contour)
        x,y = max(0,x),max(0,y)
        roi = frame[y:min(h,y+rh), x:min(w,x+rw)]
        if roi.shape[0] < 10 or roi.shape[1] < 12:
            continue
        support = np.zeros(roi.shape[:2],np.uint8)
        cv2.drawContours(support,[contour-np.array([[[x,y]]])],-1,255,-1)
        wh = cv2.inRange(cv2.cvtColor(roi,cv2.COLOR_BGR2HSV),(0,0,120),(180,75,255)) & support
        wh = cv2.morphologyEx(wh,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        whites,_ = cv2.findContours(wh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        minimum = max(24,0.015*rw*rh)
        whites = [z for z in whites if cv2.contourArea(z) >= minimum]
        if not whites:
            gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            _,wh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            wh = cv2.morphologyEx(wh & support,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            whites,_ = cv2.findContours(wh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            whites = [z for z in whites if cv2.contourArea(z) >= minimum]
        if not whites:
            continue
        white = max(whites,key=cv2.contourArea)
        approx = cv2.approxPolyDP(white,0.03*cv2.arcLength(white,True),True)
        points = approx if len(approx)==4 and cv2.isContourConvex(approx) else cv2.boxPoints(cv2.minAreaRect(white))
        q = ordered_quad(points) + np.array([x,y],np.float32)
        pw = max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]))
        ph = max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))
        # The long edge establishes the text line. Its 180-degree direction is
        # inherently ambiguous without an orientation marker; use panel_rotation.
        if ph > pw:
            q = np.roll(q,-1,axis=0)
            pw,ph = ph,pw
        turns = c.panel_rotation//90
        q = np.roll(q,-turns,axis=0)
        if turns%2:
            pw,ph = ph,pw
        pw,ph = max(12,round(pw)),max(8,round(ph))
        if pw*ph < 120:
            continue
        dst = np.array([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]],np.float32)
        H = cv2.getPerspectiveTransform(q.astype(np.float32),dst)
        if abs(np.linalg.det(H)) < 1e-12:
            continue
        panels.append((cv2.warpPerspective(frame,H,(pw,ph)),np.linalg.inv(H)))
        if len(panels) >= c.max_panels:
            break
    return panels


def select_pair(boxes, lut, c):
    candidates = []
    for a in boxes:
        if len(a) < 6 or not np.isfinite(a[:6]).all():
            continue
        x1,y1,x2,y2,conf,cls = map(float,a[:6])
        if int(cls) != cls or not 0 <= int(cls) < len(lut):
            continue
        w,h = x2-x1,y2-y1
        if w < 3 or h < 7 or w/h > 1.6 or conf < c.min_confidence:
            continue
        candidates.append((x1,y1,x2,y2,conf,lut[int(cls)]))
    candidates.sort(key=lambda a:(a[0]+a[2])/2)
    pairs=[]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            ax,ay,ar,ab,ac,ad=a
            bx,by,br,bb,bc,bd=b
            ah,bh,aw,bw=ab-ay,bb-by,ar-ax,br-bx
            if ad==0 or min(ah,bh)/max(ah,bh)<0.6:
                continue
            if abs((ay+ab)-(by+bb))/2 > 0.30*max(ah,bh):
                continue
            if min(ab,bb)-max(ay,by) < 0.60*min(ah,bh):
                continue
            if ar-bx>0.25*min(aw,bw) or bx-ar>1.2*max(ah,bh):
                continue
            number=10*ad+bd
            pixel=((min(ax,bx)+max(ar,br))/2,(min(ay,by)+max(ab,bb))/2)
            pairs.append((min(ac,bc),number,pixel))
    if not pairs:
        return None
    pairs.sort(reverse=True,key=lambda z:z[0])
    best=pairs[0]
    if any(z[1]!=best[1] and best[0]-z[0]<0.08 for z in pairs[1:]):
        return None
    return best[1],best[0],best[2]


@dataclasses.dataclass
class Observation:
    frame_id: int
    captured: float
    number: int
    confidence: float
    point: Any


def confirm_cluster(observations, now, c):
    """Complete-link spatial grouping, unique frames, and outlier isolation."""
    recent=[o for o in observations if 0<=now-o.captured<=c.confirm_window
            and np.isfinite(o.point).all() and o.confidence>=c.min_confidence]
    best=None
    for seed in recent:
        group=[seed]
        same=sorted((o for o in recent if o.number==seed.number and o is not seed),
                    key=lambda o:np.linalg.norm(o.point-seed.point))
        for o in same:
            if any(o.frame_id==g.frame_id for g in group):
                continue
            if all(np.linalg.norm(o.point-g.point)<=c.cluster_diameter for g in group):
                group.append(o)
        high=sum(o.confidence>=c.fast_confidence for o in group)
        if high<2 and len(group)<3:
            continue
        score=(len(group),float(np.mean([o.confidence for o in group])))
        if best is None or score>best[0]:
            point=np.median(np.stack([o.point for o in group]),axis=0)
            best=(score,Observation(max(o.frame_id for o in group),
                  max(o.captured for o in group),seed.number,score[1],point))
    return None if best is None else best[1]


@dataclasses.dataclass
class FlightState:
    position: Any
    velocity: Any
    q: Any
    telemetry_ok: bool
    armed: bool
    offboard: bool
    auto_land: bool
    landed: bool
    rotary: bool
    failsafe: bool
    reset_token: Any


@dataclasses.dataclass
class Directive:
    position: Any = None
    yaw: float = math.nan
    request_land: bool = False
    number: Optional[int] = None


class LandingController:
    """Pure, testable controller. It never arms or requests OFFBOARD."""

    def __init__(self,c):
        self.c=c
        self.phase="WAIT"
        self.reason=""
        self.number=None
        self.target=None
        self.seen=-math.inf
        self.command=None
        self.yaw=math.nan
        self.z_hold=None
        self.reset_token=None
        self.last_tick=None
        self.stable_since=None
        self.paused_position=None
        self.land_last=-math.inf
        self.land_attempts=0
        self.land_rejected=False

    def change(self,phase,reason=""):
        if phase!=self.phase or reason!=self.reason:
            LOG.info("phase=%s reason=%s",phase,reason)
        self.phase,self.reason=phase,reason
        self.stable_since=None

    def hold_fault(self,s,reason):
        if self.phase!="FAULT_HOLD":
            self.command=s.position.copy()
            self.yaw=yaw_of(s.q)
            self.change("FAULT_HOLD",reason)

    def slew(self,goal,s,dt):
        delta=goal-self.command
        length=float(np.linalg.norm(delta[:2]))
        if length>0:
            self.command[:2]+=delta[:2]*min(1.,self.c.horizontal_speed*dt/length)
        self.command[2]+=float(np.clip(delta[2],-self.c.descent_speed*dt,self.c.descent_speed*dt))
        # Do not let a ramp run far ahead of the actual aircraft.
        lead=self.command[:2]-s.position[:2]
        length=float(np.linalg.norm(lead))
        if length>0.6:
            self.command[:2]=s.position[:2]+lead*(0.6/length)
        self.command[2]=min(self.command[2],s.position[2]+0.20)
        return Directive(self.command.copy(),self.yaw)

    def step(self,now,s,event=None,fault="",other_publisher=False):
        dt=0.05 if self.last_tick is None else min(0.1,max(0.,now-self.last_tick))
        self.last_tick=now
        if self.phase in ("DONE","ABORTED"):
            return Directive()
        if self.phase=="AUTO_LAND":
            if s.landed or not s.armed:
                self.change("DONE","PX4 reports landed/disarmed")
            elif not s.auto_land:
                self.change("ABORTED","operator/PX4 left LAND")
            return Directive()
        if self.phase=="WAIT":
            if not (s.telemetry_ok and s.armed and s.offboard and s.rotary
                    and not s.landed and not s.failsafe) or other_publisher:
                return Directive()
            self.command=s.position.copy()
            self.yaw=yaw_of(s.q)
            self.z_hold=float(s.position[2])
            self.reset_token=s.reset_token
            self.change("SEARCH","captured fixed airborne hold point")
        # No old NED setpoint is sent when telemetry is invalid. PX4 retains
        # responsibility for its configured offboard-loss behavior.
        if s.landed or not s.armed:
            self.change("DONE","PX4 reports landed/disarmed")
            return Directive()
        if self.phase=="REQUEST_LAND" and s.auto_land:
            self.change("AUTO_LAND","PX4 entered LAND; stop offboard stream")
            return Directive()
        if not s.telemetry_ok or not s.offboard or s.failsafe or other_publisher:
            self.change("ABORTED","invalid state, mode override, failsafe, or competing publisher")
            return Directive()
        if s.reset_token!=self.reset_token:
            self.reset_token=s.reset_token
            self.command=s.position.copy()
            self.yaw=yaw_of(s.q)
            self.hold_fault(s,"EKF reset; old target coordinates invalid")
        if fault:
            self.hold_fault(s,fault)
        if self.phase=="FAULT_HOLD":
            return Directive(self.command.copy(),self.yaw)
        emitted=None
        if event is not None and event.captured>self.seen and 0<=now-event.captured<=self.c.vision_timeout:
            if self.number is None:
                if 10<=event.number<=99 and np.isfinite(event.point).all():
                    self.number=event.number
                    self.target=event.point.copy()
                    self.seen=event.captured
                    emitted=self.number
                    self.change("ALIGN","number and spatial cluster confirmed")
            elif event.number==self.number and np.linalg.norm(event.point[:2]-self.target[:2])<=self.c.target_gate:
                self.target[:2]=0.75*self.target[:2]+0.25*event.point[:2]
                self.seen=event.captured
        if self.phase=="SEARCH":
            return Directive(self.command.copy(),self.yaw)
        if self.phase=="REQUEST_LAND":
            if self.land_rejected or (self.land_attempts>=5 and now-self.land_last>=1.0):
                self.hold_fault(s,"LAND rejected or no LAND mode after five requests")
                return Directive(self.command.copy(),self.yaw)
            send=now-self.land_last>=1.0
            if send:
                self.land_last=now
                self.land_attempts+=1
            return Directive(self.command.copy(),self.yaw,send)
        fresh=now-self.seen<=self.c.vision_timeout
        if not fresh:
            if self.paused_position is None:
                self.paused_position=s.position.copy()
                self.command=self.paused_position.copy()
                self.z_hold=float(s.position[2])
                self.change("ALIGN","vision lost; hold fixed position, no descent")
            return Directive(self.paused_position.copy(),self.yaw,number=emitted)
        self.paused_position=None
        error=float(np.linalg.norm(self.target[:2]-s.position[:2]))
        speed=float(np.linalg.norm(s.velocity[:2]))
        aligned=(error<=self.c.align_radius and speed<=self.c.align_speed
                 and abs(s.velocity[2])<=self.c.align_speed)
        height=self.c.ground_z-s.position[2]
        if height<=0.05:
            self.hold_fault(s,"invalid ground height; do not command below the plane")
            return Directive(self.command.copy(),self.yaw,number=emitted)
        if self.phase=="DESCEND" and error>1.5*self.c.align_radius:
            self.z_hold=float(s.position[2])
            self.command[2]=self.z_hold
            self.change("ALIGN","horizontal error; pause descent and realign")
        if self.phase=="ALIGN":
            if aligned:
                if self.stable_since is None:
                    self.stable_since=now
                if now-self.stable_since>=self.c.align_seconds:
                    self.change("DESCEND","stable alignment")
            else:
                self.stable_since=None
        goal=np.array([self.target[0],self.target[1],self.z_hold],float)
        if self.phase=="DESCEND":
            # If already below handover height, hold current z rather than climb.
            goal[2]=max(self.c.ground_z-self.c.handover_height,self.z_hold)
            if error>self.c.align_radius or speed>self.c.align_speed:
                # Pause vertical progress while correcting even a small drift.
                self.command[2]=float(s.position[2])
                goal[2]=float(s.position[2])
            if aligned and height<=self.c.handover_height+0.10:
                if self.stable_since is None:
                    self.stable_since=now
                if now-self.stable_since>=self.c.align_seconds:
                    self.command=np.array([self.target[0],self.target[1],s.position[2]],float)
                    self.change("REQUEST_LAND","aligned low-altitude handover")
            else:
                self.stable_since=None
        d=self.slew(goal,s,dt) if self.phase!="REQUEST_LAND" else Directive(self.command.copy(),self.yaw)
        d.number=emitted
        return d


@dataclasses.dataclass
class FramePacket:
    seq: int
    message: Any
    stamp: float
    received_ros: float
    received_mono: float

    @property
    def captured_mono(self):
        return self.received_mono - (self.received_ros-self.stamp)


class Shared:
    def __init__(self):
        self.lock=threading.RLock()
        self.positions=collections.deque(maxlen=800)
        self.attitudes=collections.deque(maxlen=1200)
        self.calibration=None
        self.generation=0
        self.active=False
        self.model_ready=False
        self.last_pipeline_ok=-math.inf
        self.locked_number=None
        self.locked_point=None
        self.target=None
        self.fault=""
        self.latest=None
        self.condition=threading.Condition(self.lock)
        self.stop=threading.Event()
        self.stats=collections.Counter()
        self.last_notice={}

    def fail(self,reason):
        with self.lock:
            if not self.fault:
                self.fault=reason
                LOG.error("mission fault: %s",reason)

    def notice(self,key,message,period=5.0):
        now=time.monotonic()
        with self.lock:
            if now-self.last_notice.get(key,-math.inf)>=period:
                self.last_notice[key]=now
                LOG.warning(message)

    def put_latest(self,packet):
        with self.condition:
            if self.latest is not None:
                self.stats["frames_skipped_by_inference"]+=1
            self.latest=packet
            self.condition.notify()

    def take_latest(self):
        with self.condition:
            self.condition.wait_for(lambda:self.latest is not None or self.stop.is_set(),timeout=0.5)
            packet,self.latest=self.latest,None
            return packet


class JpegRecorder:
    """FIFO recording separate from latest-only inference; no sampling.

    A bounded queue prevents memory exhaustion. Overflow/write errors are
    counted and latch a mission fault instead of silently claiming full capture.
    Every valid delivered frame is saved while storage throughput is adequate.
    """

    def __init__(self,c,folder,shared):
        self.c,self.folder,self.shared=c,folder,shared
        self.queue=queue.Queue(maxsize=c.save_queue_frames)
        self.failed=False
        self.closed=False
        self.threads=[threading.Thread(target=self.worker,name="jpeg-%d"%i,daemon=True)
                      for i in range(c.save_workers)]
        for t in self.threads:
            t.start()

    def submit(self,packet):
        if self.failed or self.closed:
            with self.shared.lock:
                self.shared.stats["jpeg_not_saved"]+=1
            return
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.failed=True
            with self.shared.lock:
                self.shared.stats["jpeg_not_saved"]+=1
            self.shared.fail("JPEG queue overflow: recording cannot keep up; frame %d not saved"%packet.seq)

    def worker(self):
        while True:
            packet=self.queue.get()
            try:
                if packet is None:
                    return
                frame=image_to_bgr(packet.message)
                name="camera_%09d_%019d.jpg"%(packet.seq,max(0,round(packet.stamp*1e9)))
                path=self.folder/name
                if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,self.c.jpeg_quality]):
                    raise OSError("cv2.imwrite returned false: "+str(path))
                with self.shared.lock:
                    self.shared.stats["jpeg_saved"]+=1
            except Exception as exc:
                self.failed=True
                with self.shared.lock:
                    self.shared.stats["jpeg_not_saved"]+=1
                self.shared.fail("JPEG save failed: "+str(exc))
                LOG.exception("JPEG worker error")
            finally:
                self.queue.task_done()

    def close(self):
        self.closed=True
        # Called after ROS callbacks stop; drain all queued images on Ctrl+C.
        for _ in self.threads:
            self.queue.put(None)
        self.queue.join()
        for t in self.threads:
            t.join()


class VisionWorker:
    def __init__(self,c,shared):
        self.c,self.shared=c,shared
        self.thread=threading.Thread(target=self.run,name="digit-inference",daemon=True)
        self.observations=collections.deque(maxlen=80)
        self.last_frame_stamp=-math.inf
        self.generation=-1

    def start(self):
        self.thread.start()

    def load_model(self):
        # All CUDA/model operations stay on this single worker thread.
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(1)
        if not Path(self.c.model_path).is_file():
            raise FileNotFoundError("model file is missing: "+self.c.model_path)
        device,half=(0,True) if torch.cuda.is_available() else ("cpu",False)
        attempts=[(device,half)] if device=="cpu" else [(device,half),("cpu",False)]
        for i,(device,half) in enumerate(attempts):
            try:
                model=YOLO(self.c.model_path,verbose=False)
                if getattr(model,"task",None)!="detect":
                    raise ValueError("weights must be an object detection model")
                lut=class_lut(model.names,self.c.class_map)
                options=dict(conf=self.c.min_confidence,iou=0.45,imgsz=self.c.imgsz,
                             max_det=8,agnostic_nms=True,device=device,half=half,verbose=False)
                dummy=np.zeros((self.c.imgsz,self.c.imgsz,3),np.uint8)
                model.predict(dummy,**options)
                LOG.info("model ready device=%s half=%s digit_map=%s",device,half,lut)
                return model,lut,options
            except Exception:
                if i+1==len(attempts):
                    raise
                LOG.exception("CUDA initialization failed; recreate the model on CPU")
                # A new YOLO object is necessary: changing predictor args alone
                # may leave its already-built backend on the old device.
        raise RuntimeError("model initialization failed")

    def process(self,packet,model,lut,options):
        c,s=self.c,self.shared
        now=time.monotonic()
        if packet.stamp<=0 or packet.stamp<=self.last_frame_stamp:
            s.notice("image_stamp","Image timestamps missing/repeated/out of order; check camera header clock")
            return
        self.last_frame_stamp=packet.stamp
        camera_age=packet.received_ros-packet.stamp
        age=now-packet.captured_mono
        if camera_age < -0.05 or age<0 or age>c.max_frame_age:
            s.notice("image_age","Image timestamp not in ROS clock or frame too old; no vision control from this frame")
            return
        with s.lock:
            if s.fault:
                return
            active=s.active
            generation=s.generation
            positions=list(s.positions)
            attitudes=list(s.attitudes)
            info=s.calibration
            locked=s.locked_number
            locked_point=None if s.locked_point is None else s.locked_point.copy()
        if generation!=self.generation:
            self.observations.clear()
            self.generation=generation
        position=interpolate(positions,packet.stamp,c.pose_tolerance)
        q=interpolate(attitudes,packet.stamp,c.pose_tolerance,quaternion=True)
        if position is None or q is None:
            s.notice("pose_sync","No pose near image capture time; verify XRCE/ROS time synchronization")
            return
        frame=image_to_bgr(packet.message)
        calibration=scaled_calibration(c,info,frame.shape[1],frame.shape[0])
        if calibration is None:
            s.notice("calibration","No usable camera calibration")
            return
        with s.lock:
            s.last_pipeline_ok=time.monotonic()
        if not active:
            return
        panels=red_white_panels(frame,c)
        if not panels:
            return
        start=time.monotonic()
        results=model.predict([roi for roi,_ in panels],**options)
        elapsed=time.monotonic()-start
        if time.monotonic()-packet.captured_mono>c.max_frame_age:
            s.notice("inference_old","Inference result too old; reduce image/model size or use GPU")
            return
        for result,(_,inverse) in zip(results,panels):
            if result.boxes is None:
                continue
            # One device-to-host transfer instead of separate xyxy/conf/cls copies.
            boxes=result.boxes.data.detach().cpu().numpy()
            pair=select_pair(boxes,lut,c)
            if pair is None:
                continue
            number,confidence,center=pair
            if locked is not None and number!=locked:
                continue
            pixel=cv2.perspectiveTransform(np.array(center,float).reshape(1,1,2),inverse)[0,0]
            point=project_ground(pixel,position,q,calibration,c)
            if point is None:
                continue
            if locked_point is not None and np.linalg.norm(point[:2]-locked_point[:2])>c.target_gate:
                continue
            self.observations.append(Observation(packet.seq,packet.captured_mono,number,confidence,point))
        confirmed=confirm_cluster(self.observations,time.monotonic(),c)
        with s.lock:
            s.stats["inference_batches"]+=1
            s.stats["inference_ms_last"]=round(elapsed*1000,2)
            s.stats["image_to_result_ms_last"]=round((time.monotonic()-packet.captured_mono)*1000,2)
            if confirmed is not None and generation==s.generation and s.active and not s.fault:
                if s.locked_number is None or confirmed.number==s.locked_number:
                    s.target=confirmed

    def run(self):
        try:
            model,lut,options=self.load_model()
            with self.shared.lock:
                self.shared.model_ready=True
        except Exception as exc:
            self.shared.fail("model initialization failed: "+str(exc))
            LOG.exception("model initialization error")
            return
        errors=0
        while not self.shared.stop.is_set():
            packet=self.shared.take_latest()
            if packet is None:
                continue
            try:
                self.process(packet,model,lut,options)
                errors=0
            except Exception as exc:
                errors+=1
                LOG.exception("vision frame %d failed",packet.seq)
                if errors>=3:
                    self.shared.fail("three consecutive vision exceptions: "+str(exc))
                    return


def px4_topic(base,message_class,override=""):
    """Use the installed px4_msgs schema version; do not mix versions."""
    if override:
        return override
    version=int(getattr(message_class,"MESSAGE_VERSION",0))
    return base+("_v%d"%version if version else "")

