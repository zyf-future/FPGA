#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PX4 quadrotor mission: take off, scan 8x6 m at 2.5 m, land on a digit.

Run alongside vision_recognition.py. This is the only flight-command publisher.
Starting the program authorizes automatic OFFBOARD entry and normal arming,
after telemetry, visual pipeline and sole-controller checks are satisfied.
"""
from __future__ import annotations
import copy
import dataclasses
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
LOG=logging.getLogger("flight_control")
CONTEXT="/digit_mission/context"
VISION_STATUS="/digit_mission/vision_status"
TARGET="/digit_mission/target"
ROUTE_FR=((0.,0.),(8.,0.),(8.,2.),(0.,2.),(0.,4.),(8.,4.),(8.,6.),(0.,6.))


def parser():
    p=core.make_parser()
    p.description=__doc__
    p.add_argument("--altitude",type=float,default=2.5)
    p.add_argument("--scan-speed",type=float,default=0.5)
    p.add_argument("--climb-speed",type=float,default=0.4)
    p.add_argument("--waypoint-dwell",type=float,default=1.0)
    p.add_argument("--prestream-seconds",type=float,default=1.5)
    p.add_argument("--vision-heartbeat-timeout",type=float,default=1.5)
    p.add_argument("--takeoff-timeout",type=float,default=30.0)
    p.add_argument("--mission-timeout",type=float,default=300.0)
    p.add_argument("--ground-offset-m",type=float,default=0.0)
    p.add_argument("--absolute-ground-z",type=float,default=None)
    p.add_argument("--log-dir",default="./flight_logs")
    p.set_defaults(horizontal_speed=0.5)
    return p


def check_config(c):
    core.validate_config(c)
    for key in ("altitude","scan_speed","climb_speed","waypoint_dwell",
                "prestream_seconds","vision_heartbeat_timeout","takeoff_timeout","mission_timeout"):
        if not math.isfinite(getattr(c,key)) or getattr(c,key)<=0:
            raise ValueError("invalid positive parameter "+key)
    if c.prestream_seconds<1.1:
        raise ValueError("prestream must exceed one second")
    if c.altitude<=c.handover_height+0.3:
        raise ValueError("scan altitude must exceed landing handover height")
    if not math.isfinite(c.ground_offset_m) or not 0<=c.ground_offset_m<=0.8:
        raise ValueError("ground offset must be 0..0.8 m")
    if c.absolute_ground_z is not None and not math.isfinite(c.absolute_ground_z):
        raise ValueError("ground z must be finite")


def route_ned(home,yaw,ground_z,altitude):
    # Forward=[cos(yaw), sin(yaw)], right=[-sin(yaw),cos(yaw)] in NED.
    basis=np.array([[math.cos(yaw),-math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    return [np.r_[home[:2]+basis@np.array(fr),ground_z-altitude] for fr in ROUTE_FR]


@dataclasses.dataclass
class Action:
    position: object=None
    yaw: float=math.nan
    command: str=""


class Mission:
    """Pure mission state machine, independent of ROS and wall-clock sleeps."""
    def __init__(self,c):
        self.c=c
        self.id=uuid.uuid4().hex
        self.phase="WAIT_READY"
        self.reason=""
        self.home=None
        self.yaw=math.nan
        self.ground_z=None
        self.route=[]
        self.index=0
        self.sp=None
        self.reset=None
        self.last_tick=None
        self.entered=0.
        self.started=None
        self.stable=None
        self.last_request=-math.inf
        self.attempts=0
        self.command_denied=""
        self.land=None
        self.return_z=None
        self.print_number=None
        self.ground_link_seen=False

    def change(self,phase,now,reason=""):
        if self.phase!=phase or self.reason!=reason:
            LOG.info("phase=%s reason=%s",phase,reason)
        self.phase,self.reason=phase,reason
        self.entered=now
        self.stable=None
        self.last_request=-math.inf
        self.attempts=0
        self.command_denied=""

    def fixed_fault(self,now,s,reason):
        if self.phase!="FAULT_HOLD":
            self.sp=s.position.copy()
            self.yaw=core.yaw_of(s.q)
            self.change("FAULT_HOLD",now,reason)
        return Action(self.sp.copy(),self.yaw)

    def local_fr(self,p):
        delta=p[:2]-self.home[:2]
        co,si=math.cos(self.yaw),math.sin(self.yaw)
        return np.array([co*delta[0]+si*delta[1],-si*delta[0]+co*delta[1]])

    def within(self,p,margin=1.0):
        f,r=self.local_fr(p)
        return -margin<=f<=8.+margin and -margin<=r<=6.+margin

    def ramp(self,goal,s,dt,vertical_speed=None):
        d=goal-self.sp
        dist=float(np.linalg.norm(d[:2]))
        if dist>0:
            self.sp[:2]+=d[:2]*min(1.,self.c.scan_speed*dt/dist)
        rate=self.c.climb_speed if vertical_speed is None else vertical_speed
        self.sp[2]+=float(np.clip(d[2],-rate*dt,rate*dt))
        lead=self.sp[:2]-s.position[:2]
        distance=float(np.linalg.norm(lead))
        if distance>0.5:
            self.sp[:2]=s.position[:2]+lead*.5/distance
        self.sp[2]=float(np.clip(self.sp[2],s.position[2]-.25,s.position[2]+.20))
        return Action(self.sp.copy(),self.yaw)

    def arrived(self,now,s,goal,dwell=None):
        ok=(np.linalg.norm(s.position[:2]-goal[:2])<=.20
            and abs(s.position[2]-goal[2])<=.15
            and np.linalg.norm(s.velocity)<=.20)
        if not ok:
            self.stable=None
            return False
        if self.stable is None:
            self.stable=now
        return now-self.stable >= (self.c.waypoint_dwell if dwell is None else dwell)

    def request(self,now,name):
        send=""
        if now-self.last_request>=1.0:
            self.last_request=now
            self.attempts+=1
            send=name
        return Action(self.sp.copy(),self.yaw,send)

    def accepted_target(self,event,now):
        return (event is not None and 10<=event.number<=99
                and 0<=now-event.captured<=self.c.vision_timeout
                and np.isfinite(event.point).all()
                and abs(event.point[2]-self.ground_z)<.05
                and self.within(event.point,margin=1.0))

    def begin_target(self,now,s,event):
        cfg=copy.copy(self.c)
        cfg.ground_z=self.ground_z
        self.land=core.LandingController(cfg)
        self.land.step(now,s)
        d=self.land.step(now,s,event)
        self.print_number=event.number
        self.change("TARGET_LANDING",now,"confirmed target; scan route cancelled")
        return Action(d.position,d.yaw,"land" if d.request_land else "")

    def context(self):
        landing_phase=self.land.phase if self.land is not None else ""
        active=(self.phase=="SCAN" or (self.phase=="TARGET_LANDING" and
                  landing_phase in ("ALIGN","DESCEND")))
        target=None if self.land is None or self.land.target is None else self.land.target.tolist()
        return dict(schema=1,mission_id=self.id,phase=self.phase,landing_phase=landing_phase,
                    ground_z=self.ground_z,detect=active,locked_number=self.print_number,
                    target_ned=target)

    def step(self,now,s,vision_ready,event=None,competing=False,preflight=False):
        dt=.05 if self.last_tick is None else min(.10,max(0.,now-self.last_tick))
        self.last_tick=now
        if self.phase in ("DONE","ABORTED"):
            return Action()
        if self.phase=="AUTO_LAND":
            if not s.telemetry_ok:
                return Action()  # do not reclaim control from PX4
            if s.landed:
                self.change("DONE",now,"landed")
            elif not s.armed or not s.auto_land:
                self.change("ABORTED",now,"pilot/PX4 left LAND")
            return Action()
        if self.phase=="WAIT_READY":
            # Capture an unarmed, stationary ground origin; never start a new
            # automatic mission on an already flying or armed vehicle.
            if not (s.telemetry_ok and s.rotary and s.landed and not s.armed
                    and not s.failsafe and preflight and not competing):
                self.stable=None
                return Action()
            if self.home is None:
                self.home=s.position.copy()
                self.yaw=core.yaw_of(s.q)
                self.ground_z=(self.home[2]+self.c.ground_offset_m
                               if self.c.absolute_ground_z is None else self.c.absolute_ground_z)
                if not -.05<=self.ground_z-self.home[2]<=.8:
                    self.change("ABORTED",now,"ground plane inconsistent with grounded body position")
                    return Action()
                self.route=route_ned(self.home,self.yaw,self.ground_z,self.c.altitude)
                self.reset=s.reset_token
                self.sp=self.home.copy()
                LOG.info("origin=%s ground_z=%s yaw=%s route=%s",
                         self.home.tolist(),self.ground_z,self.yaw,[p.tolist() for p in self.route])
            if (s.reset_token!=self.reset or np.linalg.norm(s.position-self.home)>.2
                    or abs(math.atan2(math.sin(core.yaw_of(s.q)-self.yaw),
                                      math.cos(core.yaw_of(s.q)-self.yaw)))>.1):
                self.home=None
                self.id=uuid.uuid4().hex
                self.stable=None
                return Action()
            if not vision_ready or np.linalg.norm(s.velocity)>.15:
                self.stable=None
                return Action()
            if self.stable is None:
                self.stable=now
            if now-self.stable<.5:
                return Action()
            self.change("PRESTREAM",now,"vision and ground telemetry ready")
        if not s.telemetry_ok or s.failsafe or competing:
            self.change("ABORTED",now,"stale telemetry/failsafe/competing controller; stop publishing")
            return Action()
        if self.phase=="REQUEST_LAND" and s.auto_land:
            self.change("AUTO_LAND",now,"PX4 accepted LAND")
            return Action()
        if self.phase=="TARGET_LANDING":
            # The landing helper owns the final visual alignment and LAND ack.
            if not s.offboard and not (self.land.phase=="REQUEST_LAND" and s.auto_land):
                self.change("ABORTED",now,"operator/PX4 left OFFBOARD")
                return Action()
            if s.reset_token!=self.land.reset_token:
                # Let the landing helper re-anchor a hold in the new frame
                # before applying any checks against old mission coordinates.
                d=self.land.step(now,s)
                self.reset=s.reset_token
                self.sp=s.position.copy()
                self.yaw=core.yaw_of(s.q)
                self.change("FAULT_HOLD",now,"estimator reset during visual landing")
                return Action(self.sp.copy(),self.yaw) if s.offboard else Action()
            if s.offboard and (not self.within(s.position,margin=3.)
                    or s.position[2]<self.ground_z-self.c.altitude-1.0):
                return self.fixed_fault(now,s,"flight envelope exceeded during target approach")
            if not vision_ready and self.land.phase not in ("REQUEST_LAND","AUTO_LAND"):
                return self.fixed_fault(now,s,"vision process unhealthy; fixed hover")
            if event is not None and not self.accepted_target(event,now):
                event=None
            d=self.land.step(now,s,event)
            if self.land.phase=="AUTO_LAND":
                self.change("AUTO_LAND",now,"PX4 accepted target LAND")
            elif self.land.phase=="DONE":
                self.change("DONE",now,"landed")
            elif self.land.phase=="ABORTED":
                self.change("ABORTED",now,self.land.reason)
            elif self.land.phase=="FAULT_HOLD":
                self.sp=d.position.copy()
                self.change("FAULT_HOLD",now,self.land.reason)
            return Action(d.position,d.yaw,"land" if d.request_land else "")
        startup=self.phase in ("PRESTREAM","REQUEST_MODE","REQUEST_ARM")
        if s.reset_token!=self.reset:
            self.reset=s.reset_token
            if startup:
                self.change("ABORTED",now,"estimator reset during startup")
                return Action()
            self.sp=s.position.copy()
            self.yaw=core.yaw_of(s.q)
            return self.fixed_fault(now,s,"estimator reset invalidates route and home")
        if startup:
            if not s.landed or not preflight or not vision_ready:
                self.change("ABORTED",now,"startup readiness lost")
                return Action()
            if self.phase=="PRESTREAM":
                if s.armed:
                    self.change("ABORTED",now,"unexpected arming before startup sequence")
                    return Action()
                if now-self.entered>=self.c.prestream_seconds:
                    self.change("REQUEST_MODE",now,"request OFFBOARD after setpoint prestream")
                return Action(self.sp.copy(),self.yaw)
            if self.phase=="REQUEST_MODE":
                if s.offboard:
                    self.change("REQUEST_ARM",now,"OFFBOARD confirmed")
                    return Action(self.sp.copy(),self.yaw)
                if self.command_denied or self.attempts>=5 and now-self.last_request>=1.0:
                    self.change("ABORTED",now,"OFFBOARD rejected/timed out")
                    return Action()
                return self.request(now,"offboard")
            if not s.offboard:
                self.change("ABORTED",now,"OFFBOARD lost before arming")
                return Action()
            if s.armed:
                self.started=now
                self.change("TAKEOFF",now,"normal arming confirmed")
                return Action(self.sp.copy(),self.yaw)
            if self.command_denied or self.attempts>=5 and now-self.last_request>=1.0:
                self.change("ABORTED",now,"arming rejected/timed out")
                return Action()
            return self.request(now,"arm")
        if not s.offboard:
            self.change("ABORTED",now,"operator/PX4 left OFFBOARD")
            return Action()
        if not s.armed:
            self.change("DONE" if s.landed else "ABORTED",now,"disarmed")
            return Action()
        if self.phase=="FAULT_HOLD":
            return Action(self.sp.copy(),self.yaw)
        if s.landed and self.phase!="TAKEOFF":
            self.change("DONE",now,"landed")
            return Action()
        if not self.within(s.position,margin=3.) or s.position[2]<self.ground_z-self.c.altitude-1.0:
            return self.fixed_fault(now,s,"flight envelope exceeded")
        if not vision_ready and self.phase not in ("RETURN","HOME_DESCEND","REQUEST_LAND"):
            return self.fixed_fault(now,s,"vision heartbeat or camera pipeline lost")
        if self.started is not None and now-self.started>self.c.mission_timeout:
            if self.phase not in ("RETURN","HOME_DESCEND","REQUEST_LAND"):
                self.return_z=float(s.position[2])
                self.change("RETURN",now,"mission timeout; return at current altitude")
        if self.phase=="TAKEOFF":
            if now-self.entered>self.c.takeoff_timeout:
                return self.fixed_fault(now,s,"takeoff timeout")
            goal=self.route[0]
            d=self.ramp(goal,s,dt)
            if self.arrived(now,s,goal) and not s.landed:
                self.index=1
                self.change("SCAN",now,"A reached; start A-B-C-D-E-F-G-H")
            return d
        if self.phase=="SCAN":
            if self.accepted_target(event,now):
                return self.begin_target(now,s,event)
            goal=self.route[self.index]
            d=self.ramp(goal,s,dt)
            if self.arrived(now,s,goal):
                self.index+=1
                self.stable=None
                if self.index==len(self.route):
                    self.return_z=float(self.route[0][2])
                    self.change("RETURN",now,"scan complete; no target")
                else:
                    LOG.info("next waypoint=%s",chr(ord("A")+self.index))
            return d
        if self.phase=="RETURN":
            goal=np.r_[self.home[:2],self.return_z]
            d=self.ramp(goal,s,dt)
            if self.arrived(now,s,goal):
                self.change("HOME_DESCEND",now,"returned to takeoff horizontal position")
            return d
        if self.phase=="HOME_DESCEND":
            goal=np.r_[self.home[:2],self.ground_z-self.c.handover_height]
            if np.linalg.norm(s.position[:2]-self.home[:2])>.20:
                goal[2]=s.position[2]
                self.sp[2]=s.position[2]
            d=self.ramp(goal,s,dt,self.c.descent_speed)
            if self.arrived(now,s,np.r_[self.home[:2],self.ground_z-self.c.handover_height]):
                self.change("REQUEST_LAND",now,"home aligned; request LAND")
            return d
        if self.phase=="REQUEST_LAND":
            if self.command_denied or self.attempts>=5 and now-self.last_request>=1.:
                return self.fixed_fault(now,s,"LAND rejected/timed out")
            return self.request(now,"land")
        return Action()


def decode_target(payload,mission_id,boot_id,ros_now,mono_now,max_age):
    """Reject wrong mission/session, stale/future stamps and malformed payloads."""
    try:
        d=json.loads(payload)
        if d["schema"]!=1 or d["mission_id"]!=mission_id or d["boot_id"]!=boot_id:
            return None
        number=d["number"]
        seq=d["seq"]
        if type(number) is not int or not 10<=number<=99 or type(seq) is not int or seq<0:
            return None
        stamp=float(d["captured_ros"])
        age=ros_now-stamp
        point=np.asarray(d["ned"],float)
        confidence=float(d["confidence"])
        if not math.isfinite(age) or not 0<=age<=max_age:
            return None
        if point.shape!=(3,) or not np.isfinite(point).all() or not .60<=confidence<=1.:
            return None
        return core.Observation(seq,mono_now-age,number,confidence,point)
    except (ValueError,TypeError,KeyError,OverflowError):
        return None


def make_node(c,folder):
    from rclpy.node import Node
    from rclpy.clock import Clock,ClockType
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,HistoryPolicy
    from std_msgs.msg import String
    from px4_msgs.msg import (VehicleLocalPosition,VehicleAttitude,VehicleStatus,
        VehicleLandDetected,VehicleCommand,VehicleCommandAck,TrajectorySetpoint,OffboardControlMode)

    def qos(depth=1):
        return QoSProfile(depth=depth,history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)

    class FlightNode(Node):
        def __init__(self):
            super().__init__("digit_mission_flight")
            self.mission=Mission(c)
            self.pos=self.q=self.status=self.land=None
            self.pos_rx=self.q_rx=self.status_rx=self.land_rx=-math.inf
            self.pos_stamp=self.q_stamp=-math.inf
            self.valid=False
            self.pos_reset=None
            self.q_reset=None
            self.vision=None
            self.vision_rx=-math.inf
            self.vision_boot=None
            self.vision_restart=False
            self.event=None
            self.event_seq=-1
            self.last_graph=self.last_context=self.last_report=-math.inf
            self.last_tick=None
            self.max_gap=0.
            self.competing=False
            self.sp_topic=core.px4_topic("/fmu/in/trajectory_setpoint",TrajectorySetpoint,c.setpoint_topic)
            self.ob_topic=core.px4_topic("/fmu/in/offboard_control_mode",OffboardControlMode,c.offboard_topic)
            self.command_pub=self.create_publisher(VehicleCommand,core.px4_topic("/fmu/in/vehicle_command",VehicleCommand,c.command_topic),qos(4))
            self.sp_pub=self.create_publisher(TrajectorySetpoint,self.sp_topic,qos())
            self.ob_pub=self.create_publisher(OffboardControlMode,self.ob_topic,qos())
            self.context_pub=self.create_publisher(String,CONTEXT,qos())
            specs=[(VehicleLocalPosition,"vehicle_local_position",c.position_topic,self.on_pos),
                   (VehicleAttitude,"vehicle_attitude",c.attitude_topic,self.on_q),
                   (VehicleStatus,"vehicle_status",c.status_topic,self.on_status),
                   (VehicleLandDetected,"vehicle_land_detected",c.land_topic,self.on_land),
                   (VehicleCommandAck,"vehicle_command_ack",c.ack_topic,self.on_ack)]
            for cls,base,override,cb in specs:
                topic=core.px4_topic("/fmu/out/"+base,cls,override)
                self.create_subscription(cls,topic,cb,qos(10))
                LOG.info("subscribe %s",topic)
            self.create_subscription(String,VISION_STATUS,self.on_vision,qos())
            self.create_subscription(String,TARGET,self.on_target,qos(3))
            self.steady=Clock(clock_type=ClockType.STEADY_TIME)
            self.create_timer(.05,self.tick,clock=self.steady)
            self.create_timer(.2,self.context_tick,clock=self.steady)
            LOG.info("flight controller started; waiting for grounded/unarmed vehicle and vision")

        def ros_now(self):
            return self.get_clock().now().nanoseconds/1e9

        def fresh_stamp(self,stamp):
            age=self.ros_now()-stamp
            return stamp>0 and -.05<=age<=c.state_timeout

        def on_pos(self,msg):
            stamp=(int(getattr(msg,"timestamp_sample",0)) or int(msg.timestamp))/1e6
            if stamp<=self.pos_stamp:
                return
            p=np.array([msg.x,msg.y,msg.z],float)
            v=np.array([msg.vx,msg.vy,msg.vz],float)
            self.valid=bool(msg.xy_valid and msg.z_valid and msg.v_xy_valid and msg.v_z_valid
                            and np.isfinite(p).all() and np.isfinite(v).all())
            self.pos=(p,v)
            self.pos_rx=time.monotonic()
            self.pos_stamp=stamp
            self.pos_reset=(int(msg.xy_reset_counter),int(msg.z_reset_counter),int(msg.heading_reset_counter))

        def on_q(self,msg):
            stamp=(int(getattr(msg,"timestamp_sample",0)) or int(msg.timestamp))/1e6
            if stamp<=self.q_stamp:
                return
            try:
                self.q=core.normalized_quaternion(msg.q)
            except ValueError:
                self.q=None
                return
            self.q_rx=time.monotonic()
            self.q_stamp=stamp
            self.q_reset=int(msg.quat_reset_counter)

        def on_status(self,msg):
            self.status,self.status_rx=msg,time.monotonic()

        def on_land(self,msg):
            self.land,self.land_rx=msg,time.monotonic()

        def on_vision(self,msg):
            try:
                d=json.loads(msg.data)
                if (d["schema"]!=1 or d["mission_id"]!=self.mission.id
                        or not isinstance(d["boot_id"],str) or not d["boot_id"]):
                    return
                age=self.ros_now()-float(d["stamp"])
                if not math.isfinite(age) or not -.05<=age<=c.vision_heartbeat_timeout:
                    return
                if self.vision_boot is None or self.mission.phase=="WAIT_READY":
                    self.vision_boot=d["boot_id"]
                    self.vision_restart=False
                elif d["boot_id"]!=self.vision_boot:
                    self.vision_restart=True
                    return
                self.vision=d
                self.vision_rx=time.monotonic()
            except (KeyError,TypeError,ValueError):
                LOG.warning("invalid vision heartbeat")

        def vision_ready(self,now):
            d=self.vision
            return bool(d and d.get("mission_id")==self.mission.id
                        and d.get("ready") is True and not d.get("fault")
                        and not self.vision_restart and now-self.vision_rx<=c.vision_heartbeat_timeout)

        def on_target(self,msg):
            event=decode_target(msg.data,self.mission.id,self.vision_boot,self.ros_now(),
                                time.monotonic(),c.vision_timeout)
            if event is None or event.frame_id<=self.event_seq:
                return
            self.event=event
            self.event_seq=event.frame_id

        def on_ack(self,msg):
            if int(msg.target_system) not in (0,c.source_system) or int(msg.target_component) not in (0,c.source_component):
                return
            table={int(VehicleCommand.VEHICLE_CMD_DO_SET_MODE):"offboard",
                   int(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM):"arm",
                   int(VehicleCommand.VEHICLE_CMD_NAV_LAND):"land"}
            name=table.get(int(msg.command))
            if name is None:
                return
            result=int(msg.result)
            LOG.info("command=%s ack_result=%s",name,result)
            denied=result not in (VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED,
                                  VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS,
                                  VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED)
            expected={"REQUEST_MODE":"offboard","REQUEST_ARM":"arm","REQUEST_LAND":"land"}.get(self.mission.phase)
            if denied and name==expected:
                self.mission.command_denied=name
            if denied and name=="land" and self.mission.land is not None and self.mission.land.phase=="REQUEST_LAND":
                self.mission.land.land_rejected=True

        def state(self,now):
            st=self.status
            p,v=self.pos if self.pos is not None else (np.zeros(3),np.zeros(3))
            q=self.q if self.q is not None else np.array([1.,0,0,0])
            ok=bool(self.pos is not None and self.q is not None and st is not None
                    and self.land is not None and self.valid and now-self.pos_rx<=c.state_timeout
                    and now-self.q_rx<=c.state_timeout and now-self.status_rx<=2.
                    and now-self.land_rx<=2. and self.fresh_stamp(self.pos_stamp) and self.fresh_stamp(self.q_stamp))
            return core.FlightState(p.copy(),v.copy(),q.copy(),ok,
                bool(st is not None and st.arming_state==VehicleStatus.ARMING_STATE_ARMED),
                bool(st is not None and st.nav_state==VehicleStatus.NAVIGATION_STATE_OFFBOARD),
                bool(st is not None and st.nav_state==VehicleStatus.NAVIGATION_STATE_AUTO_LAND),
                bool(self.land is not None and self.land.landed),
                bool(st is not None and st.vehicle_type==VehicleStatus.VEHICLE_TYPE_ROTARY_WING),
                bool(st is not None and st.failsafe),(self.pos_reset,self.q_reset))

        def publish_control(self,a):
            stamp=int(self.get_clock().now().nanoseconds//1000)
            mode=OffboardControlMode()
            mode.timestamp=stamp
            for f in ("position","velocity","acceleration","attitude","body_rate","thrust_and_torque","direct_actuator"):
                if hasattr(mode,f):
                    setattr(mode,f,f=="position")
            sp=TrajectorySetpoint()
            sp.timestamp=stamp
            sp.position=[float(x) for x in a.position]
            for f in ("velocity","acceleration","jerk"):
                if hasattr(sp,f): setattr(sp,f,[math.nan]*3)
            sp.yaw=float(a.yaw)
            sp.yawspeed=math.nan
            self.ob_pub.publish(mode)
            self.sp_pub.publish(sp)

        def command(self,name):
            cmd=VehicleCommand()
            cmd.timestamp=int(self.get_clock().now().nanoseconds//1000)
            cmd.target_system=int(self.status.system_id)
            cmd.target_component=int(self.status.component_id)
            cmd.source_system=c.source_system
            cmd.source_component=c.source_component
            cmd.from_external=True
            if name=="offboard":
                cmd.command=int(VehicleCommand.VEHICLE_CMD_DO_SET_MODE)
                cmd.param1=1.
                cmd.param2=6.
            elif name=="arm":
                cmd.command=int(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM)
                cmd.param1=1.
                cmd.param2=0.  # normal arming, never force-arm
            elif name=="land":
                cmd.command=int(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                cmd.param4=cmd.param5=cmd.param6=cmd.param7=math.nan
            else:
                raise ValueError("unknown command "+name)
            self.command_pub.publish(cmd)
            LOG.info("sent %s",name)

        def context_tick(self):
            d=self.mission.context()
            d["stamp"]=self.ros_now()
            m=String()
            m.data=json.dumps(d,allow_nan=False,separators=(",",":"))
            self.context_pub.publish(m)

        def tick(self):
            now=time.monotonic()
            if self.last_tick is not None:
                self.max_gap=max(self.max_gap,(now-self.last_tick)*1000)
            self.last_tick=now
            try:
                if now-self.last_graph>=.5:
                    self.last_graph=now
                    self.competing=(self.count_publishers(self.sp_topic)>1
                        or self.count_publishers(self.ob_topic)>1
                        or self.count_publishers(CONTEXT)>1
                        or self.count_publishers(VISION_STATUS)>1)
                s=self.state(now)
                preflight=bool(self.status is not None and getattr(self.status,"pre_flight_checks_pass",False))
                a=self.mission.step(now,s,self.vision_ready(now),self.event,self.competing,preflight)
                if a.position is not None:
                    self.publish_control(a)
                if a.command:
                    self.command(a.command)
                if now-self.last_report>=2.:
                    self.last_report=now
                    snapshot=self.mission.context()
                    snapshot.update(reason=self.mission.reason,waypoint_index=self.mission.index,
                                    vision_ready=self.vision_ready(now),max_control_gap_ms=round(self.max_gap,2))
                    # Small diagnostic snapshot, never used as control input.
                    status_queue.put(snapshot)
            except Exception:
                LOG.exception("flight control callback failed")
                self.mission.change("ABORTED",now,"callback exception; stop control")

    import queue
    import threading
    status_queue=queue.SimpleQueue()
    node=FlightNode()
    def status_writer():
        while True:
            snapshot=status_queue.get()
            if snapshot is None:
                return
            try:
                path=folder/"status.tmp"
                path.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False),encoding="utf-8")
                path.replace(folder/"status.json")
            except Exception:
                LOG.exception("status write failed")
    node.writer=threading.Thread(target=status_writer,daemon=True,name="flight-status")
    node.writer.start()
    node.stop_writer=lambda:status_queue.put(None)
    return node


def main(argv=None):
    global np
    console=None
    node=None
    executor=None
    ros=None
    code=0
    try:
        console=core.QuietConsole()
        c,ros_args=parser().parse_known_args(argv)
        if ros_args and ros_args[0]!="--ros-args":
            raise ValueError("unknown arguments: "+" ".join(ros_args))
        check_config(c)
        folder=Path(c.log_dir).expanduser().resolve()/(time.strftime("flight_%Y%m%d_%H%M%S_")+str(os.getpid()))
        folder.mkdir(parents=True,exist_ok=False)
        console.redirect(folder/"run.log")
        logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
        (folder/"config.json").write_text(json.dumps(vars(c),indent=2),encoding="utf-8")
        import numpy as n
        np=core.np=n
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        ros=rclpy
        ros.init(args=ros_args)
        node=make_node(c,folder)
        executor=SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        LOG.info("operator stopped flight program; PX4 owns current-mode/loss behavior")
    except BaseException:
        import traceback
        traceback.print_exc()
        code=1
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.)
        if node is not None:
            node.stop_writer()
            node.writer.join(timeout=2.)
            node.destroy_node()
        if ros is not None and ros.ok():
            ros.shutdown()
        if console is not None:
            console.close()
    return code


if __name__=="__main__":
    raise SystemExit(main())
