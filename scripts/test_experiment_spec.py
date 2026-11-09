import unittest
from experiment_spec import curriculum_scale,group_spec,GROUPS,LOWER,UPPER
class Tests(unittest.TestCase):
 def test_stage_boundaries(self):
  for step,expected in [(0,.25),(47,.25),(48,.5),(119,.5),(120,.75),(191,.75),(192,1),(239,1),(240,1)]:
   self.assertEqual(curriculum_scale(step,240,True),expected)
 def test_control_always_full_range(self):
  for step in [0,48,120,239]:self.assertEqual(curriculum_scale(step,240,False),1)
 def test_factorial_and_budget(self):
  specs=[group_spec(g,60000,7,2048) for g in GROUPS]
  self.assertEqual({(s['curriculum'],s['randomization']) for s in specs},{(False,False),(True,False),(False,True),(True,True)})
  for s in specs:
   self.assertEqual(s['command_upper'],list(UPPER));self.assertEqual(s['command_lower'],list(LOWER))
   self.assertEqual(s['num_envs']*s['trainer_steps'],2048*60000)
 def test_invalid_progress(self):
  with self.assertRaises(ValueError):curriculum_scale(0,0,True)
if __name__=='__main__':unittest.main()
