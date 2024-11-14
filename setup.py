from setuptools import setup

setup(
   name='uitb-tools',
   version='0.2.0',
   author='Florian Fischer',
   author_email='florian.fischer26@gmail.com',
   packages=['uitb_evaluate', 'uitb_reach_envelope', 'uitb_reward_scaling'],
   url='https://github.com/fl0fischer/uitb-tools',
   license='LICENSE',
   description='Evaluation, Analysis and Visualization Tools for Biomechanical Simulation in MuJoCo',
   long_description=open('README.md').read(),
   python_requires='>=3.8',
   install_requires=[
       # "gym", #>=0.26.0",
       # "mujoco>=2.2.0",
       "numpy", "matplotlib", "scipy", "pandas", "seaborn"
   ],
)
