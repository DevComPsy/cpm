"""Sphinx configuration for the cpm documentation."""

import datetime
import inspect
import sys
from pathlib import Path

import cpm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "_ext"))

# -- Project information -----------------------------------------------------

project = "cpm"
author = "the cpm developers"
copyright = f"2024-{datetime.date.today().year}, {author}"
release = version = cpm.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "sphinx_reredirects",
    "sphinx_sitemap",
    "notfound.extension",
    "cpm_docs",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "jupyter_execute",
    "scripts",
    "_ext",
    "**/_code",
    "**/.ipynb_checkpoints",
]
default_role = "literal"

# -- API reference -----------------------------------------------------------

autosummary_generate = True
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "none"
autoclass_content = "class"
add_module_names = False

# Tutorials and examples shown in a "See also" box on the API page of each
# object, rendered by _templates/autosummary/class.rst.
API_TUTORIAL_LINKS = {
    "cpm.generators.Parameters": [
        "get-started/concepts/parameters",
        "tutorials/first-model",
    ],
    "cpm.generators.Value": ["get-started/concepts/parameters"],
    "cpm.generators.Wrapper": ["get-started/concepts/models", "tutorials/first-model"],
    "cpm.generators.Simulator": [
        "tutorials/parameter-recovery",
        "tutorials/model-recovery",
    ],
    "cpm.optimisation.FminBound": [
        "tutorials/first-model",
        "tutorials/parameter-recovery",
    ],
    "cpm.optimisation.compare.PenalisedLikelihoods": ["tutorials/model-recovery"],
    "cpm.hierarchical.EmpiricalBayes": ["tutorials/hierarchical-empirical-bayes"],
    "cpm.hierarchical.VariationalBayes": [
        "tutorials/hierarchical-variational-bayes"
    ],
    "cpm.applications.reinforcement_learning.RLRW": [
        "get-started/quickstart",
        "tutorials/first-model",
    ],
    "cpm.applications.reinforcement_learning.HybridMBMF": ["examples/two-step-task"],
    "cpm.models.learning.SARSATrace": ["examples/two-step-task"],
    "cpm.models.learning.SeparableRule": ["examples/blocking"],
    "cpm.models.learning.DeltaRule": ["examples/blocking", "tutorials/first-model"],
    "cpm.applications.signal_detection.EstimatorMetaD": ["examples/meta-d"],
    "cpm.datasets.load_bandit_data": ["tutorials/first-model"],
    "cpm.datasets.load_metacognition_data": ["examples/meta-d"],
    "cpm.datasets.load_two_step_data": ["examples/two-step-task"],
}
autosummary_context = {"tutorial_links": API_TUTORIAL_LINKS}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = "all"
numpydoc_xref_aliases = {
    "pd.DataFrame": "pandas.DataFrame",
    "DataFrame": "pandas.DataFrame",
    "pd.Series": "pandas.Series",
    "np.ndarray": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
    "Parameters": "cpm.generators.Parameters",
    "Wrapper": "cpm.generators.Wrapper",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Notebooks and MyST ------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3

# Notebooks (.ipynb) are rendered from their stored outputs; the MyST text
# notebooks of the concept pages are executed on every build.
nb_execution_mode = "force"
nb_execution_excludepatterns = ["*.ipynb"]
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_merge_streams = True
nb_mime_priority_overrides = [
    ("html", "application/vnd.jupyter.widget-view+json", None),
    ("html", "application/vnd.microsoft.datawrangler.viewer.v0+json", None),
]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"cpm {version}"
html_baseurl = "https://devcompsy.github.io/cpm/"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/logo/favicon.png"
html_sourcelink_suffix = ""
html_show_sourcelink = False
html_sidebars = {"index": [], "about/citing": []}
html_context = {
    "github_user": "DevComPsy",
    "github_repo": "cpm",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
html_theme_options = {
    "logo": {
        "image_light": "_static/logo/cpm-mark.png",
        "image_dark": "_static/logo/cpm-mark-dark.png",
        "text": "cpm",
        "alt_text": "cpm - home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/DevComPsy/cpm",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cpm-toolbox/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "cpm-toolbox.net",
            "url": "https://cpm-toolbox.net",
            "icon": "fa-solid fa-globe",
        },
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_prev_next": True,
    # Pages not listed here get the theme default: page-toc, edit-this-page, sourcelink.
    "secondary_sidebar_items": {
        "tutorials/*": ["page-toc", "notebook-links", "edit-this-page"],
        "examples/*": ["page-toc", "notebook-links", "edit-this-page"],
        "get-started/quickstart": ["page-toc", "notebook-links", "edit-this-page"],
        "api/generated/*": ["page-toc"],
    },
    "announcement": (
        "cpm is published in <i>PLOS Computational Biology</i>. If you use it, "
        "please <a href='https://devcompsy.github.io/cpm/about/citing.html'>"
        "cite Dome et al. (2026)</a>."
    ),
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True

codeautolink_concat_default = True
codeautolink_autodoc_inject = False

sitemap_url_scheme = "{link}"
notfound_urls_prefix = "/cpm/"
linkcheck_allowed_redirects = {r"https://doi\.org/.*": r".*"}

# -- Redirects from the old MkDocs site --------------------------------------

_LEGACY_PAGES = {
    "installation": "get-started/installation",
    "troubleshooting": "get-started/troubleshooting",
    "roadmap": "about/roadmap",
    "contributing": "about/contributing",
    "examples/bandit-task": "tutorials/first-model",
    "examples/model-parameter-recovery": "tutorials/parameter-recovery",
    "examples/model-recovery": "tutorials/model-recovery",
    "examples/model-recovery-run": "tutorials/model-recovery",
    "examples/functions/__init__": "tutorials/model-recovery",
    "examples/functions/model_anti": "tutorials/model-recovery",
    "examples/functions/model_delta": "tutorials/model-recovery",
    "examples/functions/model_kernel": "tutorials/model-recovery",
    "examples/fitting-hierarchical-estimation": "tutorials/hierarchical-empirical-bayes",
    "examples/fitting-hierarchical": "tutorials/hierarchical-variational-bayes",
    "examples/fitting-hierarchical-run": "tutorials/hierarchical-variational-bayes",
    "examples/associative-learning": "examples/blocking",
    "examples/metacognition": "examples/meta-d",
    "examples/hpc-example": "how-to/run-on-hpc",
    "examples/example6": "examples/index",
    **{
        f"api/{m}": f"api/{m}"
        for m in [
            "applications",
            "datasets",
            "generators",
            "hierarchical",
            "models",
            "optimisation",
            "utils",
        ]
    },
}
# MkDocs served every page as a folder (old/index.html); the new pages are
# old.html, so each legacy folder gets an index.html that redirects.
redirects = {
    f"{old}/index": "../" * (old.count("/") + 1) + f"{new}.html"
    for old, new in _LEGACY_PAGES.items()
}

# -- Source links ------------------------------------------------------------


def linkcode_resolve(domain, info):
    """Link each API object to its source on GitHub."""
    if domain != "py" or not info.get("module"):
        return None
    obj = sys.modules.get(info["module"])
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
    try:
        obj = inspect.unwrap(obj)
        filename = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None
    if filename is None:
        return None
    root = Path(cpm.__file__).resolve().parents[1]
    rel = Path(filename).resolve().relative_to(root).as_posix()
    ref = "main" if "dev" in release else release
    end = start + len(lines) - 1
    return f"https://github.com/DevComPsy/cpm/blob/{ref}/{rel}#L{start}-L{end}"
