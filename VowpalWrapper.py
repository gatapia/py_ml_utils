from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base
import numpy as np

class _VW(sklearn.base.BaseEstimator):

    """scikit-learn interface for Vowpal Wabbit

    Only works for regression and binary classification.
    """

    def __init__(self,
                 logger=None,
                 vw='vw',
                 moniker='moniker',
                 name=None,
                 bits=None,
                 loss=None,
                 passes=10,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 working_dir=None,
                 decay_learning_rate=None,
                 initial_t=None,
                 minibatch=None,
                 total=None,
                 node=None,
                 unique_id=None,
                 span_server=None,
                 bfgs=None,
                 oaa=None,
                 old_model=None,
                 incremental=False,
                 mem=None,
                 nn=None,
                 ):
        self.logger = logger
        self.vw = vw
        self.moniker = moniker
        self.name = name
        self.bits = bits
        self.loss = loss
        self.passes = passes
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.quadratic = quadratic
        self.audit = audit
        self.power_t = power_t
        self.adaptive = adaptive
        self.working_dir = working_dir
        self.decay_learning_rate = decay_learning_rate
        self.initial_t = initial_t
        self.minibatch = minibatch
        self.total = total
        self.node = node
        self.unique_id = unique_id
        self.span_server = span_server
        self.bfgs = bfgs
        self.oaa = oaa
        self.old_model = old_model
        self.incremental = incremental
        self.mem = mem
        self.nn = nn

    def fit(self, X, y):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: [{<feature name>: <feature value>}]
            input features
        y: [int or float]
            output labels
        """
        examples = X

        # initialize model
        self.vw_ = VW(
            logger=self.logger,
            vw=self.vw,
            moniker=self.moniker,
            name=self.name,
            bits=self.bits,
            loss=self.loss,
            passes=self.passes,
            log_stderr_to_file=self.log_stderr_to_file,
            silent=self.silent,
            l1=self.l1,
            l2=self.l2,
            learning_rate=self.learning_rate,
            quadratic=self.quadratic,
            audit=self.audit,
            power_t=self.power_t,
            adaptive=self.adaptive,
            working_dir=self.working_dir,
            decay_learning_rate=self.decay_learning_rate,
            initial_t=self.initial_t,
            minibatch=self.minibatch,
            total=self.total,
            node=self.node,
            unique_id=self.unique_id,
            span_server=self.span_server,
            bfgs=self.bfgs,
            oaa=self.oaa,
            old_model=self.old_model,
            incremental=self.incremental,
            mem=self.mem,
            nn=self.nn
        )

        # add examples to model
        self.vw_.training(examples)

        # learning done after "with" statement
        return self

    def predict(self, X):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: [{<feature name>: <feature value>}]
            input features
        """
        examples = X

        # add test examples to model
        self.vw_.predicting(examples)

        # read out predictions
        predictions = np.asarray(list(self.vw_.read_predictions_()))

        return predictions


class VW_Regressor(sklearn.base.RegressorMixin, _VW):
    pass


class VW_Classifier(sklearn.base.ClassifierMixin, _VW):

    def predict(self, X):
        result = super(VW_Classifier, self).predict(X)
        return result


class VW:
    def __init__(self,
                 logger=None,
                 vw='vw',
                 moniker=None,
                 name=None,
                 bits=None,
                 loss=None,
                 passes=None,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 cubic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 working_dir=None,
                 decay_learning_rate=None,
                 initial_t=None,
                 lda=None,
                 lda_D=None,
                 lda_rho=None,
                 lda_alpha=None,
                 minibatch=None,
                 total=None,
                 node=None,
                 unique_id=None,
                 span_server=None,
                 bfgs=None,
                 oaa=None,
                 old_model=None,
                 incremental=False,
                 mem=None,
                 nn=None,
                 holdout_off=None,
                 no_model=None,
                 **kwargs):
        assert moniker and passes
        
        if logger is None:
            self.log = VPLogger()
        else:
            self.log = logger

        self.node = node
        self.total = total
        self.unique_id = unique_id
        self.span_server = span_server
        if self.node is not None:
            assert self.total is not None
            assert self.unique_id is not None
            assert self.span_server is not None

        if name is None:
            self.handle = '%s' % moniker
        else:
            self.handle = '%s.%s' % (moniker, name)

        if self.node is not None:
            self.handle = "%s.%d" % (self.handle, self.node)

        if old_model is None:
            self.filename = '%s.model' % self.handle
            self.incremental = False
        else:
            self.filename = old_model
            self.incremental = True

        self.name = name
        self.bits = bits
        self.loss = loss
        self.vw = vw
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.passes = passes
        self.quadratic = quadratic
        self.cubic = cubic
        self.power_t = power_t
        self.adaptive = adaptive
        self.decay_learning_rate = decay_learning_rate
        self.audit = audit
        self.initial_t = initial_t
        self.lda = lda
        self.lda_D = lda_D
        self.lda_rho = lda_rho
        self.lda_alpha = lda_alpha
        self.minibatch = minibatch
        self.oaa = oaa
        self.bfgs = bfgs
        self.mem = mem
        self.nn = nn
        self.holdout_off = holdout_off
        self.no_model = no_model
        
        self.tmpdir = 'tmpfiles'
        if not os.path.isdir(self.tmpdir): os.mkdir(self.tmpdir)

        # Do some sanity checking for compatability between models
        if self.lda:
            assert not self.l1
            assert not self.l1
            assert not self.l2
            assert not self.loss
            assert not self.adaptive
            assert not self.oaa
            assert not self.bfgs
        else:
            assert not self.lda_D
            assert not self.lda_rho
            assert not self.lda_alpha
            assert not self.minibatch

        self.working_directory = working_dir or os.getcwd()

    def vw_base_command(self, base):
        l = base
        if self.no_model            is     None: l.append('-f %s' % self.get_model_file())
        if self.bits                is not None: l.append('-b %d' % self.bits)
        if self.learning_rate       is not None: l.append('--learning_rate=%f' % self.learning_rate)
        if self.l1                  is not None: l.append('--l1=%f' % self.l1)
        if self.l2                  is not None: l.append('--l2=%f' % self.l2)
        if self.initial_t           is not None: l.append('--initial_t=%f' % self.initial_t)
        if self.quadratic           is not None: l.append('-q %s' % self.quadratic)
        if self.cubic               is not None: l.append('--cubic %s' % self.cubic)
        if self.power_t             is not None: l.append('--power_t=%f' % self.power_t)
        if self.loss                is not None: l.append('--loss_function=%s' % self.loss)
        if self.decay_learning_rate is not None: l.append('--decay_learning_rate=%f' % self.decay_learning_rate)
        if self.lda                 is not None: l.append('--lda=%d' % self.lda)
        if self.lda_D               is not None: l.append('--lda_D=%d' % self.lda_D)
        if self.lda_rho             is not None: l.append('--lda_rho=%f' % self.lda_rho)
        if self.lda_alpha           is not None: l.append('--lda_alpha=%f' % self.lda_alpha)
        if self.minibatch           is not None: l.append('--minibatch=%d' % self.minibatch)
        if self.oaa                 is not None: l.append('--oaa=%d' % self.oaa)
        if self.unique_id           is not None: l.append('--unique_id=%d' % self.unique_id)
        if self.total               is not None: l.append('--total=%d' % self.total)
        if self.node                is not None: l.append('--node=%d' % self.node)
        if self.span_server         is not None: l.append('--span_server=%s' % self.span_server)
        if self.mem                 is not None: l.append('--mem=%d' % self.mem)
        if self.audit:                           l.append('--audit')
        if self.bfgs:                            l.append('--bfgs')
        if self.adaptive:                        l.append('--adaptive')
        if self.nn                  is not None: l.append('--nn=%d' % self.nn)
        if self.holdout_off         is not None: l.append('--holdout_off')
        return ' '.join(l)

    def vw_train_command(self, cache_file):
        if os.path.exists(self.get_model_file()) and self.incremental:
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -i %s' \
                % (self.passes, cache_file, self.get_model_file())
        else:
            self.log.debug('No existing model file or not options.incremental')
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s' \
                    % (self.passes, cache_file)

    def vw_test_command(self, model_file, prediction_file):
        return self.vw_base_command([self.vw]) + ' -t -i %s -r %s' % (model_file, prediction_file)

    def vw_test_command_library(self, model_file):
        return self.vw_base_command([]) + ' -t -i %s' % (model_file)

    def training(self, instances):        
        f = self.save_tmp_file(instances, True)
        self.start_training(f)
        self.close_process()
        # self.delete_tmp_file(f)

    def predicting(self, instances):
        f = self.save_tmp_file(instances, False)
        self.start_predicting(f)
        self.close_process()

    def save_tmp_file(self, instances, training=True):        
        f, _ = self.tmpfile('_tmp_' + ('training' if training else 'testing') + '_file.vw.')
        with open(f, 'wb') as fs: 
            fs.write('\n'.join(instances))
        os.close(_)
        return f

    def tmpfile(self, suffix):
        _, f = tempfile.mkstemp(dir=self.tmpdir, suffix=suffix)
        return (self.tmpdir + '/' + f.split('\\')[-1], _)


    def predicting_library(self):
        self.start_predicting_library()
        self.end_predicting_library()

    def start_training(self, training_file):
        cache_file = self.get_cache_file()
        model_file = self.get_model_file()

        # Remove the old cache and model files
        if not self.incremental:
            safe_remove(cache_file)
            safe_remove(model_file)

        # Run the actual training
        self.vw_process = self.make_subprocess(
            self.vw_train_command(cache_file), training_file)

    def close_process(self):
        # Close the process
        assert self.vw_process
        if self.vw_process.wait() != 0:
            raise Exception("vw_process %d (%s) exited abnormally with return code %d" % \
                (self.vw_process.pid, self.vw_process.command, self.vw_process.returncode))

    def start_predicting(self, testing_file):
        model_file = self.get_model_file()
        # Be sure that the prediction file has a unique filename, since many processes may try to
        # make predictions using the same model at the same time
        pred_file = self.get_prediction_file()
        prediction_file, _ = self.tmpfile(pred_file)
        os.close(_)

        self.vw_process = self.make_subprocess(
            self.vw_test_command(model_file, prediction_file), testing_file)
        self.prediction_file = prediction_file        

    def start_predicting_library(self):
        import vw_py
        model_file = self.get_model_file()
        self.vw_process = vw_py.VW(self.vw_test_command_library(model_file))

    def end_predicting_library(self):
        # Close the process
        assert self.vw_process
        self.vw_process.finish()

    def parse_prediction(self, p):
        if self.lda:
            return map(float, p.split())
        else:
            return float(p.split()[0])

    def read_predictions_(self):
        for x in open(self.prediction_file):
            yield self.parse_prediction(x)
        # clean up the prediction file
        os.remove(self.prediction_file)


    def make_subprocess(self, command, file):
        if not self.log_stderr_to_file:
            stdout = open('/dev/null', 'w')
            stderr = open('/dev/null', 'w') if self.silent else sys.stderr
            self.current_stdout = None
            self.current_stderr = None
        else:
            # Save the output of vw to file for debugging purposes
            log_file_base = tempfile.mktemp(dir=self.working_directory, prefix="vw-")
            self.current_stdout = log_file_base + '.out'
            self.current_stderr = log_file_base + '.err'
            stdout = open(self.current_stdout, 'w')
            stderr = open(self.current_stderr, 'w')
            stdout.write(command + '\n')
            stderr.write(command + '\n')
        self.log.debug('Running command: "%s"' % str(command))
        commands = shlex.split(str(command))
        commands.append(file)
        result = subprocess.Popen(commands, 
            stdout=stdout, stderr=stderr, 
            close_fds=sys.platform != "win32", 
            universal_newlines=True, cwd='.')
        result.command = command
        return result

    def get_current_stdout(self):
        return open(self.current_stdout)

    def get_current_stderr(self):
        return open(self.current_stderr)

    def get_model_file(self):
        if self.incremental:
            return self.filename
        else:
            return self.tmpdir + '/' + self.filename

    def get_cache_file(self):
        return self.tmpdir + '/' +  '%s.cache' % (self.handle)

    def get_prediction_file(self):
        return '%s.prediction' % (self.handle)

def safe_remove(f):
    try: os.remove(f)
    except OSError:
        pass

class VPLogger:
    def debug(self, s):
        print '[DEBUG] %s' % s

    def info(self, s):
        print '[INFO] %s' % s

    def warning(self, s):
        print '[WARNING] %s' % s

    def error(self, s):
        print '[ERROR] %s' % s
