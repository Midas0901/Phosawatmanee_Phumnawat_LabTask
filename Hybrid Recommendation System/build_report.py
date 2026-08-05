from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8, alignment=TA_LEFT)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=10, spaceBefore=2)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=6, spaceBefore=12)
meta = ParagraphStyle("meta", parent=styles["Normal"], fontSize=9.5, textColor="#555555", spaceAfter=14)

story = []
story.append(Paragraph("Milestone 2 - Initial EDA &amp; Progress Report", h1))
story.append(Paragraph("IAT 461 - Final Project | Hybrid Recommendation System for Asian Drama Content (DramaDome)", meta))

story.append(Paragraph(
    "For this milestone I focused on getting familiar with both datasets, cleaning up the parts that "
    "were going to cause problems later, and figuring out roughly how I want to approach the actual "
    "recommendation problem before I start building anything more complicated. Below is basically the "
    "order I worked through things, plus what I found along the way.", body))

story.append(Paragraph("1. What I did first: loading and checking the data", h2))
story.append(Paragraph(
    "I started by just loading dramas.csv and users.csv into pandas and looking at the shape and dtypes "
    "before touching anything else. dramas.csv has 5,000 rows and 33 columns, and users.csv has 10,000 "
    "rows with 6 columns (user_id, gender, location, user_watch_history, preferred_genre, preferred_country). "
    "My plan was to get a column dictionary written down first so I don't forget what half these fields mean "
    "later - things like no_of_viewers, rank, and popularity look similar but aren't the same thing, so I "
    "want to be careful not to mix them up when I build the ranking logic.", body))

story.append(Paragraph("2. Checking for missing values and obvious data quality issues", h2))
story.append(Paragraph(
    "Next I ran .isna().sum() across dramas.csv to see what's actually usable. A few things stood out:", body))
story.append(ListFlowable([
    ListItem(Paragraph("NEGATIVE_people_sentiment and POSITIVE_people_sentiment are missing for roughly "
                        "44-59% of rows, so I don't think I can rely on the sentiment columns as a core "
                        "feature - I'll treat them as optional/supplementary at best.", body)),
    ListItem(Paragraph("screenwriter (1,480 missing) and director (1,103 missing) have a lot of gaps too, "
                        "so any \"find dramas by this director\" feature will only work for part of the catalogue.", body)),
    ListItem(Paragraph("tags is only missing for 64 rows and genres has effectively no missing values, so "
                        "those two are my most reliable structured fields and will probably do most of the "
                        "heavy lifting in the matching logic.", body)),
    ListItem(Paragraph("rating and no_of_viewers are missing for the same ~15 rows (4,985 out of 5,000 have "
                        "both), so I'll likely just drop those 15 rather than try to impute them.", body)),
], bulletType="bullet", start="circle"))

story.append(Paragraph(
    "I also checked for duplicate titles (none found by exact name match, which was a relief) and confirmed "
    "there's no overlap issue joining on the country field - country only takes 7 distinct values "
    "(South Korea, China, Japan, Thailand, Taiwan, Philippines, Hong Kong), with South Korea and China making "
    "up more than half the catalogue between them. That's good to know because it means \"preferred_country\" "
    "filtering is going to be pretty coarse - it's not evenly spread.", body))

story.append(Paragraph("3. Checking the users.csv side and how it connects to dramas.csv", h2))
story.append(Paragraph(
    "Since users.csv is the synthetic dataset, I wanted to actually verify the watch histories line up with "
    "real titles in dramas.csv rather than just trusting it. I parsed the user_watch_history field (it's stored "
    "as a JSON-looking string) and checked what fraction of watched titles I can actually find in dramas.csv by "
    "name - it came back at 100% for the sample I checked, so the join key is clean and I don't need a fuzzy-matching "
    "step there, which saves me some work.", body))

story.append(Paragraph("4. Deciding on an approach: why this is a pattern-discovery problem, not a prediction problem", h2))
story.append(Paragraph(
    "The client proposal (DramaDome) asks for a ranked shortlist of dramas given a user's structured profile "
    "plus a free-text description of what they want to watch. I went back and forth on this, but I don't think "
    "there's an actual label to predict here - there's no \"did the user like this recommendation\" column, "
    "and no historical click/rating signal tying a specific user to a specific drama outcome. What we do have "
    "is content: genres, tags, cast, synopsis text, and the user's stated preferences. So my plan is to treat "
    "this as a content-based / retrieval problem rather than supervised classification: use TF-IDF (or similar) "
    "over the tag/genre/synopsis text to represent each drama, then score dramas against a combination of "
    "(a) the user's structured preferences (genre, country match) and (b) semantic similarity to their free-text "
    "query. I'll validate this with things like silhouette score if I end up clustering titles by genre/tag "
    "similarity, but I'm treating that as exploratory rather than a strict pass/fail metric, per the assignment "
    "guidance.", body))

story.append(Paragraph("5. Assumptions I'm making so far", h2))
story.append(ListFlowable([
    ListItem(Paragraph("Treating rows with missing rating/no_of_viewers (15 rows) as droppable rather than "
                        "trying to impute, since it's under 0.5% of the catalogue.", body)),
    ListItem(Paragraph("Treating genres and tags as the primary structured signal, and sentiment columns as "
                        "a \"nice to have\" rather than something the core system depends on, given how sparse "
                        "they are.", body)),
    ListItem(Paragraph("Assuming exact (case-insensitive) name matching is good enough to join watch history "
                        "to the catalogue, since the 100% match check on the sample held up.", body)),
], bulletType="bullet", start="circle"))

story.append(Paragraph("6. What's next", h2))
story.append(Paragraph(
    "For the next checkpoint I want to build out the actual scoring function that combines structured "
    "preference matching with text similarity, get it running end-to-end on a few test queries, and start "
    "thinking about how I'll explain/justify individual recommendations back to the user (e.g. showing which "
    "genre or tag drove a match), since the client proposal specifically asked for explainability, not just "
    "a ranked list.", body))

doc = SimpleDocTemplate("/mnt/user-data/outputs/Milestone2_Progress_Report.pdf",
                         pagesize=letter,
                         topMargin=0.75*inch, bottomMargin=0.75*inch,
                         leftMargin=0.85*inch, rightMargin=0.85*inch)
doc.build(story)
print("done")
