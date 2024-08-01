import numpy as np

class Neural_standalone(object):

  def __init__(self, number_controls, weight=1e-4, **kwargs):
    self._nu = number_controls
    self._ctrl = np.zeros((self._nu,)) 
    self._weight = weight

  def cost(self):
    return self._weight * np.sum(self._ctrl ** 2)

  def reset(self):
    pass

  def update(self, ctrl):
    self._ctrl = ctrl.copy()


class CumulativeFatigue3CCr_standalone(object):

  # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles -- using control signals
  # instead of torque or force
  # v2: now with additional muscle fatigue state
  def __init__(self, number_activations, dt, weight=5e-4):
    self._na = number_activations
    self._r = 7.5
    self._F = 0.0146
    self._R = 0.0022
    self._LD = 10
    self._LR = 10
    self._MA = None
    self._MR = None
    self._MF = None
    self._TL = None
    self._dt = dt
    self._weight = weight
    self._effort_cost = None
    
    self.reset()

  def cost(self):
    # Calculate effort
    effort = np.linalg.norm(self._MA - self._TL)
    self._effort_cost = self._weight*effort
    # self._effort_cost = self._weight*(effort**2)  #VARIANT
    return self._effort_cost

  def reset(self):
    self._MA = np.zeros((self._na,))
    self._MR = np.ones((self._na,))
    self._MF = np.zeros((self._na,))

  def update(self, act):
    # Get target load
    TL = act.copy()
    self._TL = TL

    # Calculate C(t)
    C = np.zeros_like(self._MA)
    idxs = (self._MA < TL) & (self._MR > (TL - self._MA))
    C[idxs] = self._LD * (TL[idxs] - self._MA[idxs])
    idxs = (self._MA < TL) & (self._MR <= (TL - self._MA))
    C[idxs] = self._LD * self._MR[idxs]
    idxs = self._MA >= TL
    C[idxs] = self._LR * (TL[idxs] - self._MA[idxs])

    # Calculate rR
    rR = np.zeros_like(self._MA)
    idxs = self._MA >= TL
    rR[idxs] = self._r*self._R
    idxs = self._MA < TL
    rR[idxs] = self._R

    # Clip C(t) if needed, to ensure that MA, MR, and MF remain between 0 and 1
    C = np.clip(C, np.maximum(-self._MA/self._dt + self._F*self._MA, (self._MR - 1)/self._dt + rR*self._MF),
                np.minimum((1 - self._MA)/self._dt + self._F*self._MA, self._MR/self._dt + rR*self._MF))

    # Update MA, MR, MF
    dMA = (C - self._F*self._MA)*self._dt
    dMR = (-C + rR*self._MF)*self._dt
    dMF = (self._F*self._MA - rR*self._MF)*self._dt
    self._MA += dMA
    self._MR += dMR
    self._MF += dMF
    
  def _get_state(self):
    state = {"3CCr_MA": self._MA,
         "3CCr_MR": self._MR,
         "3CCr_MF": self._MF,
         "effort_cost": self._effort_cost}
    return state

class ConsumedEndurance_acts_standalone(object):

  lifting_muscles = ["DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed"]
  lifting_indices = [0, 1, 2, 3, 4, 5, 6, 20, 21, 15, 16, 17]
  maximum_shoulder_torques = np.array([1218.9, 1103.5,  201.6,  499.2, 1075.8, 1306.9,  269.5,  525.1,
        316.8,  771.8,  717.5,  717.5])
  
  # consumed endurance model, taken from https://dl.acm.org/doi/pdf/10.1145/2556288.2557130
  def __init__(self, number_activations, dt, weight=0.01):
    self._na = number_activations
    self._dt = dt
    self._weight = weight
    self._endurance = None
    self._consumed_endurance = None
    self._effort_cost = None
    
  def get_endurance(self, actuator_force):
    #applied_shoulder_torque = np.linalg.norm(data.qfrc_inverse[:])
    #applied_shoulder_torque = np.linalg.norm(data.qfrc_actuator[:])
    
    applied_shoulder_torques = actuator_force[self.lifting_indices]
    # self.maximum_shoulder_torques /= self.maximum_shoulder_torques  #TODO: delete this, if actual forces are used!
    
    #assert np.all(applied_shoulder_torque <= 0), "Expected only negative values in data.actuator_force."
    #strength = np.mean((applied_shoulder_torques/self.maximum_shoulder_torques)**2)
    strength = np.abs(applied_shoulder_torques/self.maximum_shoulder_torques)  #compute strength per muscle
    # assert np.all(strength <= 1), f"Applied torque is larger than maximum torque! strength:{strength}, applied:{applied_shoulder_torques}, max:{maximum_shoulder_torques}"
    strength = strength.clip(0, 1)
    
    # if strength > 0.15:
    #     endurance = (1236.5/((strength*100 - 15)**0.618)) - 72.5
    # else:
    #     endurance = np.inf
    
    endurance = np.inf * np.ones_like(strength)
    endurance[strength > 0.15] = (1236.5/((strength[strength > 0.15]*100 - 15)**0.618)) - 72.5
    
    minimum_endurance = np.min(endurance)
    # TODO: take minimum of each muscle synergy, and then apply sum/mean
    
    return minimum_endurance
    
  def cost(self):
    #total_time = data.time
    consumed_time = self._dt
    
    if self._endurance < np.inf:
        self._consumed_endurance = (consumed_time/self._endurance)*100
    else:
        self._consumed_endurance = 0.0
    
    self._effort_cost = self._weight*self._consumed_endurance
    return self._effort_cost

  def reset(self):
    pass

  def update(self, act):
    # Calculate consumed endurance
    self._endurance = self.get_endurance(act)

  def _get_state(self):
    state = {"consumed_endurance": self._consumed_endurance,
             "effort_cost": self._effort_cost}
    return state

    