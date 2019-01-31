package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "bhge_bayesianGP",
    srcs = ["__init__.py"],
    deps = [
        ":bayesiangp",
        ":bgpmodel",
    ],
)

py_library(
    name = "bayesiangp",
    srcs = ["bayesiangp.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "bgpmodel",
    srcs = ["bgpmodel.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        ":bayesiangp",
    ],
)
