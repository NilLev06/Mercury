"""Privileged bicycle teacher. Never imported by student inference."""
import heapq
import itertools
import math
from dataclasses import replace
from ..sim.kinematics import BicycleState,step_bicycle
from .arrival import body_goal_distance


class BicycleTeacher:
    def __init__(self, env):
        self.env = env
        self.path = self._plan()
        self.cursor = 0
        self.planned_pose = env.car_pose()
        self.escape = None

    def _command_safe(self, throttle, steering):
        state = self.env.state
        for _ in range(5):
            state = step_bicycle(state,throttle,steering,self.env.drive,self.env.drive.timestep)
            if self.env._pose_collides(state):
                return False
        return True

    def _safe(self, state, margin=.045):
        # The environment's exact chassis + outer wheel footprint, with a
        # modest lateral tracking margin. Goal sphere is not an obstacle.
        for lateral in (-margin,0,margin):
            shifted = BicycleState(state.x-math.sin(state.yaw)*lateral,
                                   state.y+math.cos(state.yaw)*lateral,state.yaw)
            if self.env._pose_collides(shifted):
                return False
        return True

    def _plan(self):
        env = self.env
        start = env.car_pose()
        # A student's legal contact pose can already violate the optional
        # tracking margin. Requiring that margin on the first escape sample
        # traps the planner. Recovery still checks the full wheel footprint.
        margin = .045 if self._safe(BicycleState(*start)) else 0.0
        initial_state = env.state
        def initial_safe(state,direction,steering):
            for _ in range(5):
                state = step_bicycle(state,.7*direction,steering,env.drive,env.drive.timestep)
                if not self._safe(state,margin):
                    return False
            return True
        if not any(initial_safe(initial_state,d,s)
                   for d in (1,-1) for s in (-1.,-.5,0.,.5,1.)):
            # Momentum may make every immediate command collide. Plan the
            # escape after braking, without changing the environment; action()
            # still checks actual momentum and issues its braking fallback.
            initial_state = replace(initial_state,speed=0.,yaw_rate=0.)
        goal = env.goal
        def distance(p):
            return math.hypot(goal[0]-p[0]-env.drive.wheelbase/2*math.cos(p[2]),
                              goal[1]-p[1]-env.drive.wheelbase/2*math.sin(p[2]))
        def key(p):
            return round(p[0]/.10),round(p[1]/.10),round(p[2]/(math.pi/18))%36
        queue, counter = [],itertools.count()
        heapq.heappush(queue,(distance(start),next(counter),0.0,start,()))
        costs = {key(start):0.0}
        for _ in range(16000):
            if not queue:
                break
            _,_,cost,pose,path = heapq.heappop(queue)
            if cost > costs.get(key(pose),math.inf)+1e-9:
                continue
            if distance(pose) <= .39:
                return list(path)
            for direction in (1,-1):
                for steering in (-1.0,-.5,0.0,.5,1.0):
                    # The first coarse waypoint can lie beyond a grazing
                    # collision. Check the actual acceleration/control interval
                    # before admitting an initial escape primitive.
                    if not path:
                        if not initial_safe(initial_state,direction,steering):
                            continue
                    x,y,yaw = pose
                    samples = []
                    curvature = math.tan(steering*env.drive.max_steer_rad)/env.drive.wheelbase
                    for _ in range(8):
                        ds = direction*.025
                        middle = yaw+curvature*ds/2
                        x += ds*math.cos(middle)
                        y += ds*math.sin(middle)
                        yaw = (yaw+curvature*ds+math.pi)%(2*math.pi)-math.pi
                        state = BicycleState(x,y,yaw)
                        if not self._safe(state,margin):
                            break
                        samples.append((x,y,yaw,direction,steering))
                    if len(samples) != 8:
                        continue
                    new = x,y,yaw
                    new_cost = cost+.2*(1 if direction==1 else 2.5)+.01*abs(steering)
                    k = key(new)
                    if new_cost >= costs.get(k,math.inf):
                        continue
                    costs[k] = new_cost
                    heapq.heappush(queue,(new_cost+max(0,distance(new)-.39),next(counter),
                                          new_cost,new,path+tuple(samples)))
        raise ValueError('Bicycle teacher could not find a footprint-safe route')

    def action(self):
        env = self.env
        distance = body_goal_distance(env)
        if distance <= .43:
            return 0.0,0.0
        x,y,yaw = env.car_pose()
        if self.escape is not None:
            origin,command = self.escape
            if math.hypot(x-origin[0],y-origin[1]) < .15 and self._command_safe(*command):
                return command
            self.escape = None
            self.path,self.cursor = self._plan(),0
            self.planned_pose = env.car_pose()
        window = self.path[self.cursor:min(len(self.path),self.cursor+24)]
        deviation = min((math.hypot(p[0]-x,p[1]-y) for p in window),default=math.inf)
        exhausted = self.cursor >= len(self.path)-2
        px,py,pyaw = self.planned_pose
        moved = math.hypot(x-px,y-py)>.01 or abs((yaw-pyaw+math.pi)%(2*math.pi)-math.pi)>.05
        # A blocked student can issue hundreds of actions at the same pose.
        # Its recovery route remains valid; do not rerun A* for every label.
        if (env._contact_active and moved) or deviation > .25 or exhausted:
            self.path = self._plan()
            self.cursor = 0
            self.planned_pose = env.car_pose()
        # Follow the nearest local path point, retaining direction changes.
        end = min(len(self.path),self.cursor+24)
        # Never jump over a reversing cusp to a nearby returning waypoint.
        # Complete the current direction segment before selecting the next.
        direction = self.path[self.cursor][3]
        cusp = next((i for i in range(self.cursor+1,end)
                     if self.path[i][3]!=direction),end)
        end = cusp
        if cusp < len(self.path) and math.hypot(self.path[cusp-1][0]-x,
                                               self.path[cusp-1][1]-y)<.035:
            self.cursor = cusp
            end = min(len(self.path),self.cursor+24)
            direction = self.path[self.cursor][3]
            end = next((i for i in range(self.cursor+1,end)
                        if self.path[i][3]!=direction),end)
        self.cursor = min(range(self.cursor,end),
                          key=lambda i: math.hypot(self.path[i][0]-x,self.path[i][1]-y))
        direction = self.path[self.cursor][3]
        look = self.cursor
        while look+1 < len(self.path) and look < self.cursor+12 and self.path[look+1][3]==direction:
            look += 1
        px,py,_,_,_ = self.path[look]
        angle = (math.atan2(py-y,px-x)-yaw+math.pi)%(2*math.pi)-math.pi
        length = max(.1,math.hypot(px-x,py-y))
        steering = math.atan(2*env.drive.wheelbase*math.sin(angle)/length)/env.drive.max_steer_rad
        # Brake early enough to settle inside the same scoring radius.
        throttle = direction*min(.7,max(.12,(distance-.43)*2))
        command = throttle,max(-1,min(1,steering))
        if not self._command_safe(*command):
            # A geometric A* route is not a guarantee that pure pursuit can
            # execute it at a grazing contact. Use a checked short backoff,
            # then plan from the new pose instead of teaching repeated impacts.
            for steer in (0.,-.5,.5,-1.,1.):
                escape = -.7*direction,steer
                if self._command_safe(*escape):
                    self.escape = ((x,y),escape)
                    return escape
            # Current forward momentum may require a blocked braking tick
            # before a reverse command becomes feasible.
            return -.7*direction,0.
        return command
