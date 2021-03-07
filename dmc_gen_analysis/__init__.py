import inspect
import os
from functools import reduce
from os.path import dirname, abspath, basename

import yaml
from ml_logger import pJoin
from params_proto.neo_proto import ParamsProto, Accumulant
from termcolor import cprint

with open(pJoin(dirname(__file__), ".yours"), 'r') as stream:
    rc = yaml.load(stream, Loader=yaml.BaseLoader)


class RUN(ParamsProto):
    """The main point of this config object is to provide a way for config functions
    to directly specify the job prefix."""

    server = "http://54.71.92.65:8080"
    username = rc.get('username', None)
    project = rc.get('project', None)

    prefix = "{username}/{project}/{now:%Y/%m-%d}/{file_stem}/{job_name}"

    job_name = "{job_prefix}/{job_postfix}"
    job_prefix = '{now:%H.%M.%S}'
    job_postfix = '{job_counter}'
    job_counter = Accumulant(None)

    readme = None

    # noinspection PyMissingConstructor
    @classmethod
    def __init__(cls, job_counter=True, **kwargs):
        cls._update(**kwargs)

        if job_counter is None:
            pass
        elif job_counter is False:
            cls.job_counter = None
        elif cls.job_counter is None:
            cls.job_counter = 0
        # fuck python -- bool is subtype of int. Fuck guido.
        elif isinstance(job_counter, int) and not isinstance(job_counter, bool):
            cls.job_counter = job_counter
        else:
            cls.job_counter += 1

        cls.job_prefix = pJoin(*cls.job_prefix.format(**vars(cls)).split('/'))
        cls.job_postfix = pJoin(*cls.job_postfix.format(**vars(cls)).split('/'))
        cls.job_name = pJoin(*cls.job_name.format(**vars(cls)).split('/'))
        cls.prefix = pJoin(*cls.prefix.format(**vars(cls)).split('/'))


# if __name__ == '__main__':
#     RUN(job_counter=None)
#     assert RUN.job_counter is None
#
#     RUN(job_counter=True)
#     assert RUN.job_counter is 0
#
#     RUN(job_counter=True)
#     assert RUN.job_counter == 1
#
#     RUN(job_counter=True)
#     assert RUN.job_counter == 2
#
#     RUN(job_counter=0)
#     assert RUN.job_counter == 0
#
#     RUN(job_counter=10)
#     assert RUN.job_counter == 10


def dir_prefix(depth=-1):
    from ml_logger import logger

    caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
    # note: for scripts in the `plan2vec` module this also works -- b/c we truncate fixed depth.
    script_path = logger.truncate(caller_script, depth=len(__file__.split('/')) - 1)
    prefix = pJoin(RUN.prefix, script_path)
    return reduce(lambda p, i: dirname(p), range(-depth), prefix)


def config_charts(config_yaml, path=".charts.yml"):
    from textwrap import dedent
    from ml_logger import logger

    if not config_yaml:

        try:
            caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
            cwd = dirname(caller_script)
        except:
            cwd = os.getcwd()

    logger.log_text(dedent(config_yaml).lstrip(), path)


def instr(fn, *ARGS,
          _prefix=None, _job_name=None, _job_prefix=None, _job_postfix=None,
          _job_counter=True,
          _job_readme=None, _run_readme=None, __file=False, __silent=False, **KWARGS):
    """
    thunk for configuring the logger. The reason why this is not a decorator is

    :param fn: function to be called
    :param *ARGS: position arguments for the call
    :param _job_prefix: prefix for this job
    :param _job_postfix: postfix for this job
    :param _job_counter:

        example:
            - _job_counter == 0: sets counter to 0
            - _job_counter == None: does not use counter in logging prefix.
            - _job_counter = True: increment counter by "1" with each job.

    :param __file__: console mode, by-pass file related logging
    :param __silent: do not print
    :param **KWARGS: keyword arguments for the call
    :return: a thunk that can be called without parameters
    """
    from ml_logger import logger

    if __file:
        caller_script = pJoin(os.getcwd(), __file)
    else:
        launch_module = inspect.getmodule(inspect.stack()[1][0])
        __file = launch_module.__file__
        caller_script = abspath(__file)

    # note: for scripts in the `plan2vec` module this also works -- b/c we truncate fixed depth.
    script_path = logger.truncate(caller_script, depth=len(__file__.split('/')) - 1)
    file_stem = logger.stem(script_path)
    file_name = basename(file_stem)

    RUN(file_name=file_name, file_stem=file_stem, now=logger.now(),
        **{k[1:]: v for k, v in locals().items() if k.startswith("_job") and v})

    PREFIX = RUN.prefix

    # todo: there should be a better way to log these.
    # todo: we shouldn't need to log to the same directory, and the directory for the run shouldn't be fixed.
    logger.configure(root_dir=RUN.server, prefix=PREFIX, asynchronous=False,  # use sync logger
                     max_workers=4, register_experiment=False)
    logger.upload_file(caller_script)
    # the tension is in between creation vs run. Code snapshot are shared, but runs need to be unique.
    _ = dict()
    if ARGS:
        _['args'] = ARGS
    if KWARGS:
        _['kwargs'] = KWARGS

    logger.log_params(
        run=logger.run_info(status="created", script_path=script_path),
        revision=logger.rev_info(),
        fn=logger.fn_info(fn),
        **_,
        silent=__silent)

    logger.print('taking diff, if this step takes too long, check if your '
                 'uncommitted changes are too large.', color="green")
    logger.diff()
    if RUN.readme:
        logger.log_text(RUN.readme, "README.md", dedent=True)

    import jaynes  # now set the job name to prefix
    if jaynes.RUN.config and jaynes.RUN.mode != "local":
        runner_class, runner_args = jaynes.RUN.config['runner']
        if 'name' in runner_args:  # ssh mode does not have 'name'.
            runner_args['name'] = pJoin(file_name, RUN.job_name)
        del logger, jaynes, runner_args, runner_class
        if not __file:
            cprint(f'Set up job name', "green")

    def thunk(*args, **kwargs):
        import traceback
        from ml_logger import logger

        print(PREFIX)

        assert not (args and ARGS), f"can not use position argument at " \
                                    f"both thunk creation as well as run.\n" \
                                    f"_args: {args}\nARGS: {ARGS}"

        logger.configure(root_dir=RUN.server, prefix=PREFIX, register_experiment=False, max_workers=10)
        logger.log_params(host=dict(hostname=logger.hostname),
                          run=dict(status="running", startTime=logger.now(), job_id=logger.job_id))

        import time
        try:
            _KWARGS = {**KWARGS}
            _KWARGS.update(**kwargs)

            results = fn(*(args or ARGS), **_KWARGS)

            logger.log_line("========== execution is complete ==========")
            logger.log_params(run=dict(status="completed", completeTime=logger.now()))
            logger.flush()
            time.sleep(3)
        except Exception as e:
            tb = traceback.format_exc()
            with logger.SyncContext():  # Make sure uploaded finished before termination.
                logger.print(tb, color="red")
                logger.log_text(tb, filename="traceback.err")
                logger.print(f"{logger.hostname}: {os.environ.get('GPU_DEVICE_ORDINAL', 'N/A')}")
                logger.log_params(run=dict(status="error", exitTime=logger.now()))
                logger.flush()
            time.sleep(3)
            raise e

        return results

    return thunk
