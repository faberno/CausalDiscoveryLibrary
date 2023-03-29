import os
import warnings
import fileinput
import subprocess
import uuid
from shutil import copy, rmtree
from pathlib import Path
from tempfile import gettempdir
import cdt.utils.Settings


def launch_R_script(template, arguments, output_function=None,
                    verbose=True, debug=False):
    """Launch an R script, starting from a template and replacing text in file
    before execution.

    Args:
        template (str): path to the template of the R script
        arguments (dict): Arguments that modify the template's placeholders
            with arguments
        output_function (function): Function to execute **after** the execution
            of the R script, and its output is returned by this function. Used
            traditionally as a function to retrieve the results of the
            execution.
        verbose (bool): Sets the verbosity of the R subprocess.
        debug (bool): If True, the generated scripts are not deleted.

    Return:
        Returns the output of the ``output_function`` if not `None`
        else `True` or `False` depending on whether the execution was
        successful.
    """
    base_dir = Path('{0!s}/cdt_R_script_{1!s}'.format(gettempdir(), uuid.uuid4()))
    os.makedirs(base_dir)
    rpath = cdt.utils.Settings.SETTINGS.get_default(rpath=None)
    scriptpath = Path('{}/instance_{}'.format(base_dir, os.path.basename(template)))
    copy(str(template), str(scriptpath))

    # Converting Paths to OS-compliant paths
    for arg in arguments:
        if isinstance(arguments[arg], (Path, str)):
            arguments[arg] = str(arguments[arg]).replace('\\', '\\\\')

    with fileinput.FileInput(str(scriptpath), inplace=True) as file:
        for line in file:
            mline = line
            for elt in arguments:
                mline = mline.replace(elt, arguments[elt])
            print(mline, end='')

    if output_function is None:
        try:
            output = subprocess.call([str(rpath), "--no-restore --no-save --no-site-file", str(scriptpath)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception as e:
            print("R Call errored, is R available ?")
            raise e

    else:
        try:
            if verbose:
                process = subprocess.Popen([str(rpath), "--no-restore --no-save --no-site-file", str(scriptpath)])
            else:
                process = subprocess.Popen([str(rpath), "--no-restore --no-save --no-site-file", str(scriptpath)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
        except KeyboardInterrupt:
            if not debug:
                rmtree(base_dir)
            raise KeyboardInterrupt
        try:
            output = output_function()
        # Cleaning up
        except Exception as e:
            if not debug:
                rmtree(base_dir)
            if not verbose:
                out, err = process.communicate()
                print("\nR Python Error Output \n-----------------------\n")
                print(e)
                raise RuntimeError("RProcessError \nR Process Error Output \n-----------------------\n" + str(err, "ISO-8859-1")) from None
            print("\nR Python Error Output \n-----------------------\n")
            print(e)
            raise RuntimeError("RProcessError ") from None

    if not debug:
        rmtree(base_dir)
    return output