from document_to_podcast.cli import document_to_podcast

EXAMPLE_INPUT = """
### Mozilla's approach to trustworthy AI

Mozilla has a rich history of reimagining computing norms to favor openness and innovation. We first did this in the early 2000s by championing an openness in an era where the web was on the brink of being monopolized by a single company, Microsoft, which had gone from being a minor player in the browser market to near-dominance. The market dominance of Internet Explorer threatened to lock in users, stamp out competitors, and stifle innovation online.

In the face of Microsoft's monopolization of the browser market, a loose coalition of open source activists, software developers, and web enthusiasts came together to build

standards-based browsers and web servers that would eventually wrest power away from the

tech giant. Mozilla was an early and active member of this movement. We focused resources, coordinated code, and ultimately released Firefox as part of this movement. Around the same time, the US Department of Justice's antitrust case against Microsoft demonstrated how regulators can help keep the technology industry competitive and healthy.

The result was a fundamental shift in the computing environment of the time. A renewed interest in web standards like HTML and JavaScript made true cross-platform applications the norm, replacing the dominant paradigm of end user apps that only worked on Windows. This fostered an open environment that allowed new cross platform products and services — including Facebook and Gmail — to enter the field. The internet we know now would not exist if the constrained environment of Windows and Internet Explorer 6 had become the status quo.

Today, we are at a similar inflection point. As in the early 2000s, many of our current problems are caused by a limited playing field. There are bright spots: A growing number of software developers, activists, academics, designers, and technologists are asking critical questions about how current norms around AI and data are centralizing power, stifling innovation, and eliminating user agency. But these efforts desperately need more fuel.

In this paper, we provide Mozilla's perspective on how we might do just this. Our work began in earnest in 2019, when members of the Mozilla community began asking questions like: What can Mozilla do to shift norms around AI? Who else is tackling this problem? And, how can we help them? We emerged from the exploration process with big-picture learnings. For instance, while many of the challenges with AI are individual, large scale AI also presents major collective risks. We also emerged with granular learnings. For instance, there is progress being made in creating privacy-preserving ways to handle data for machine learning. In addition, governments are hungry to figure out how to fairly and effectively regulate AI, but they lack the internal expertise and independent research needed to do so.

All of these learnings culminated in Mozilla's theory of change — a rough road map for what levers we need to pull in order to achieve trustworthy AI at scale and in a lasting way. Some of these levers exist in the realm of industry: Mozilla can support better education for computer science students or push for greater algorithmic accountability. Some of these levers exist in policy: We can steer more like-minded technologists toward government or advocate for stricter enforcement of privacy laws. Other levers exist in civil society, in the realms of academia, activism, art, and journalism.

All these levers are interconnected, and over the coming years, Mozilla will focus our effort and resources on pulling these levers. However, we know that our own contribution to this work exists within a much larger constellation of actors. Just like we did in the early Firefox era, Mozilla will function as one part of a broader movement: focusing resources, coordinating work,

and nurturing a more equitable computing environment.
"""


def test_document_to_podcast(tmp_path):
    input_file = tmp_path / "input_file.md"
    input_file.write_text(EXAMPLE_INPUT)
    document_to_podcast(
        input_file=str(input_file), output_folder=str(tmp_path / "output")
    )
    assert (tmp_path / "output" / "podcast.txt").exists()
    assert (tmp_path / "output" / "podcast.wav").exists()
