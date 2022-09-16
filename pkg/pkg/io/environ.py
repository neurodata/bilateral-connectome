import os


def cast_to_bool(var):
    if var == "True":
        var = True
    elif var == "False":
        var = False
    return var


def get_environment_variables():
    RESAVE_DATA = cast_to_bool(os.environ.get("RESAVE_DATA", False))
    RERUN_SIMS = cast_to_bool(os.environ.get("RERUN_SIMS", False))
    DISPLAY_FIGS = cast_to_bool(os.environ.get("DISPLAY_FIGS", True))
    return RESAVE_DATA, RERUN_SIMS, DISPLAY_FIGS
