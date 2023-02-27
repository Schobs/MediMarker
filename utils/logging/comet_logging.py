import dominate
from dominate.tags import *
from dominate.util import raw


def save_comet_html(summary_results, individual_results):
    # Create tab links
    tab_names = []
    tab_summaries = []
    tab_inds = []
    for key, summary_dict in summary_results.items():
        tab_names.append(key)
        tab_summaries.append(summary_dict.to_html())
        tab_inds.append(individual_results[key].to_html())

    doc = dominate.document(title='Results')

    with doc.head:
        script(
            raw('''
            function openTab(tabName) {
                var i;
                var x = document.getElementsByClassName("tabs");
                for (i = 0; i < x.length; i++) {
                    x[i].style.display = "none";
                }
                document.getElementById(tabName).style.display = "block";
                }
            ''')
        )

    with doc:

        # first add buttons:
        for idx, i in enumerate(tab_names):
            fnc = "openTab(\'"+i+"\')"
            button(i, onclick=fnc)

        for idx, i in enumerate(tab_names):

            if idx == 0:
                with div(id=i, cls="tabs"):
                    p("Summary Results")
                    raw(tab_summaries[idx])
                    p("Individual Results")

                    raw(tab_inds[idx])
            else:
                with div(id=i, cls="tabs", style="display:none"):
                    p("Summary Results")
                    raw(tab_summaries[idx])
                    p("Individual Results")
                    raw(tab_inds[idx])

            # #tabs content
            # # tab_content = ul(cls="tabs")
            # with ul(cls="tabs-content"):
            #     # with div(id='header').add(ul(), cls="tabs"):
            #     for idx, i in enumerate(tab_summaries):
            #         if idx==0:
            #             li(i, tab_inds[idx], cls="active-tab")
            #         else:
            #             li( i, tab_inds[idx])

    return str(doc)


def comet_log_yaml_parameters(writer, yaml_parameters):
    """Logs the yaml parameters to CometML"""

    for key, value in yaml_parameters.items():
        for key2, value2 in value.items():
            writer.log_parameters({key+"_"+key2: value2})
        # logger.info("Logged yaml parameters to CometML.")
