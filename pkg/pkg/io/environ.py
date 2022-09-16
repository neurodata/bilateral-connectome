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

    print("Environment variables:")
    print("   RESAVE_DATA:", RESAVE_DATA)
    print("   RERUN_SIMS:", RERUN_SIMS)
    print("   DISPLAY_FIGS:", DISPLAY_FIGS)
    print()

    return RESAVE_DATA, RERUN_SIMS, DISPLAY_FIGS
