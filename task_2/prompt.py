import pandas as pd

LABELS = ['Self-direction thought', 'Self-direction action', 'Stimulation', 'Hedonism', 'Achievement', 'Power dominance', 'Power resources', 'Face', 'Security personal', 'Security societal', 'Tradition',
          'Conformity rules', 'Conformity interpersonal', 'Humility', 'Benevolence caring', 'Benevolence dependability', 'Universalism concern', 'Universalism nature', 'Universalism tolerance', 'Universalism objectivity']
PROMPT_FORMATS = ["The argument: '{}' \n. Question: Which value category does the argument belong to? Options: {} \n"]

def single_shot_prompt(df):
    """Creates a single shot prompt for each argument with the first prompt format"""

    template = PROMPT_FORMATS[0]  # use the first template
    prompts = [
        template.format(row['body'], ', '.join(LABELS))
        for _, row in df.iterrows()
    ]
    df['body'] = prompts
    return df


def remove_line_breaks(df):
    """Removes line breaks from the arguments"""
    df['body'] = df['body'].str.replace('\n', ' ')
    return df

if __name__ == "__main__":
    # load data
    df = pd.read_csv('subreddit_threads.csv')

    df = remove_line_breaks(df)

    # create single shot prompt
    df = single_shot_prompt(df)

    # save data
    df.to_csv('subreddit_threads_prompt.csv', index=False)  