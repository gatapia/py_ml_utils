from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base
import numpy as np
import pandas as pd
import math
from pandas_extensions import * 

class ExeEstimator(sklearn.base.BaseEstimator):
  def __init__(self):
    self.tmpdir = 'tmpfiles'
    if not os.path.isdir(self.tmpdir): os.mkdir(self.tmpdir)


  def save_tmp_file(self, instances, suffix, training=True):    
    f = self.tmpfile('_tmp_' + ('training' if training else 'testing') + '_' + suffix)
    with open(f, 'wb') as fs: fs.write('\n'.join(instances))    
    return f

  def tmpfile(self, suffix):
    _, f = tempfile.mkstemp(dir=self.tmpdir, suffix=suffix)
    os.close(_)
    return self.tmpdir + '/' + f.split('\\')[-1]

  def close_process(self, running_process):
    assert running_process

    if running_process.wait() != 0:
      raise Exception("process %d (%s) exited abnormally with return code %d" % \
        (running_process.pid, running_process.command, running_process.returncode))

  def read_predictions(self, predictions_file):
    lines = []
    with open(predictions_file) as f:
        lines=map(float, map(lambda l: l.split('#')[0], 
            f.readlines()))
    os.remove(predictions_file)
    return lines

  def make_subprocess(self, command):    
    stdout = open('nul', 'w')
    stderr = sys.stderr

    print 'running command: "%s"' % str(command)
    commands = shlex.split(str(command))
    result = subprocess.Popen(commands, 
        stdout=stdout, stderr=stderr, 
        close_fds=sys.platform != "win32", 
        universal_newlines=True, cwd='.')
    result.command = command
    return result
