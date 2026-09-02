import textwrap

class Explainable:

  """
  Mixin giving every activation/cost class a shared `.explain()`. Content
  lives in the three class attributes below; this base class only handles
  formatting, so adding a new activation/cost just means setting those
  three strings, not writing a new print routine.
  """

  _formula = ""
  _why_used = ""
  _why_not = ""

  @classmethod
  def explain(cls):
    width = 64
    bar = "=" * width
    wrap = lambda text: textwrap.fill(text, width=width)

    print(bar)
    print(cls.__name__)
    print(bar)
    print("Formula:")
    print(textwrap.indent(wrap(cls._formula), "  "))
    print()
    print("Why it's used:")
    print(textwrap.indent(wrap(cls._why_used), "  "))
    print()
    print("Why it's avoided:")
    print(textwrap.indent(wrap(cls._why_not), "  "))
    print(bar)
