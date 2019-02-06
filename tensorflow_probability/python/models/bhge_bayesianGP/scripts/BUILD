

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)


py_library(
    name = "scripts",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bayesiangp",
        ":bgpmodel",
    ],
)


py_library(
    name = "bgpmodel",
    srcs = ["bgpmodel.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bayesiangp",
    ],
)


py_library(
    name = "bayesiangp",
    srcs = ["bayesiangp.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/mcmc",
    ],
)
