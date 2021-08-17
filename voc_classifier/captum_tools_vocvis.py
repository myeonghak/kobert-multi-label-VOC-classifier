from IPython.core.display import HTML, display
HAS_IPYTHON = True

def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_tooltip(item, text):
    return '<div class="tooltip">{item}\
        <span class="tooltiptext">{text}</span>\
        </div>'.format(
        item=item, text=text
    )
def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)
def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)
min_len=64
def format_word_importances(words, importances):
    try:
        if importances is None or len(importances) == 0:
            return "<td></td>"

        if len(words) > len(importances):
            words=words[:min_len]


        tags = ["<td>"]
        for word, importance in zip(words, importances[: len(words)]):
            word = format_special_tokens(word)
            color = _get_color(importance)
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>'.format(
                color=color, word=word
            )
            tags.append(unwrapped_tag)
        tags.append("</td>")
        return "".join(tags)
    except Exception as e:
        print("skip it", e)


def visualize_text(datarecords, legend=True):
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    cnt=0
    for datarecord in datarecords:
        cnt+=1
        try:
            rows.append(
                "".join(
                    [
                        "<tr>",
                        format_classname(datarecord.true_class),
                        format_classname(
                            "{0} ({1:.2f})".format(
                                datarecord.pred_class, datarecord.pred_prob
                            )
                        ),
                        format_classname(datarecord.attr_class),
                        format_classname("{0:.2f}".format(datarecord.attr_score)),
                        format_word_importances(
                            datarecord.raw_input, datarecord.word_attributions
                        ),
                        "<tr>",
                    ]
                )
            )
        except Exception as e:
            print(f"Error in {cnt}",e)

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    display(HTML("".join(dom)))
