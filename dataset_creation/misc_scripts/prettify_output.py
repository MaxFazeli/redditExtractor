import codecs
import io
import json

encoding = "utf-8"

def reprocess_merged_file(subreddit, type):
    print("Starting to reprocess merged file of {} for {}".format(type, subreddit))
    f = io.open("{}/{}/{}_{}.txt".format(type, subreddit, subreddit, type), "r", encoding=encoding)
    g = io.open("{}/{}/{}_{}_reprocessed.txt".format(type, subreddit, subreddit, type), "w", encoding=encoding)
    for l in f:
        if "}{" in l:
            print("Merge error...")
            g.write(l.replace("}{", "}\n{"))
        else:
            g.write(l)
    f.close()
    g.close()

def process_submissions(subreddit):
    print("Starting to process Submissions of {}".format(subreddit))
    f = io.open("submissions/{}/{}_submissions.txt".format(subreddit, subreddit), "r", encoding=encoding)
    g = io.open("submissions/{}/{}_submissions_pretty.txt".format(subreddit, subreddit), "w", encoding=encoding)
    #g.write(u"subredditId\tsubreddit\tcreated_at\tauthor\tnum_comments\turl\tscore\tupvotes\tlink_flair_css_class\ttitle\tselftext\tlink\n")
    g.write(u"subredditId\tsubreddit\tcreated_at\tauthor\tnum_comments\turl\tscore\tlink_flair_css_class\ttitle\tselftext\tlink\n")
    for l in f:
        if l.strip() == "":
            continue
        #print(l)
        jl = json.loads(l.strip())
        #g.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(jl['subreddit_id'], jl['subreddit'],
        g.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(jl['subreddit_id'], jl['subreddit'],
            #jl['created_utc'], jl['author'], jl['num_comments'], jl['url'], jl['score'], jl['ups'],
            jl['created_utc'], jl['author'], jl['num_comments'], jl['url'], jl['score'],
            jl['link_flair_css_class'], jl['title'].replace("\t", " ").replace("\n", " "),
            jl['selftext'].replace("\t", " ").replace("\n", " "), jl['permalink']))
    f.close()
    g.close()
    print("Done")


def process_comments(subreddit):
    print("Starting to process Comments of {}".format(subreddit))
    f = io.open("comments/{}/{}_comments.txt".format(subreddit, subreddit), "r", encoding=encoding)
    g = io.open("comments/{}/{}_comments_pretty.txt".format(subreddit, subreddit), "w", encoding=encoding)
    g.write(u"subredditId\tsubreddit\tid\tlink_id\tname\tparent_id\tcreated_utc\tauthor\tups\tdowns\tscore\tcontroversiality\tgilded\tbody\tarchived\n")
    for l in f:
        if l.strip() == "":
            continue
        #print(l)
        jl = json.loads(l.strip())
        if "name" not in jl: jl["name"] = None
        if "downs" not in jl: jl["downs"] = None
        if "ups" not in jl: jl["ups"] = None
        if "archived" not in jl: jl["archived"] = None
        g.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(jl['subreddit_id'],
            jl['subreddit'], jl['id'], jl['link_id'], jl['name'], jl['parent_id'], jl['created_utc'], jl['author'],
            jl['ups'], jl['downs'], jl['score'], jl['controversiality'], jl['gilded'],
            jl['body'].replace("\t", " ").replace("\n", " "), jl['archived']))
    f.close()
    g.close()
    print("Done")


def main():
    subreddits = ["ifyoulikeblank"]
    for subreddit in subreddits:
        #reprocess_merged_file(subreddit, "comments")
        process_submissions(subreddit)
        process_comments(subreddit)

if __name__ == "__main__":
    main()
