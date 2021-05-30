
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    class_dict = {'lab':[], 'project':[], 'midterm':[], 'final':[], 'disc':[], 'checkpoint':[]}
   
    for col in grades.columns:
        if 'lab' in col and '-' not in col:
            class_dict['lab'].append(col)
        elif 'project' in col and '-' not in col and '_' not in col:
            class_dict['project'].append(col)
        elif 'midterm' in col and '-' not in col:
            class_dict['midterm'].append(col)
        elif 'Final' in col and '-' not in col:
            class_dict['final'].append(col)
        elif 'discussion' in col and '-' not in col:
            class_dict['disc'].append(col)
        elif 'checkpoint' in col and '-' not in col:
            class_dict['checkpoint'].append(col)
    
    return class_dict

# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    grades = grades.fillna(0)
    
    overall_grades = []
    
    project_count = len(get_assignment_names(grades)['project'])
    
    max_grades = grades[list(filter(lambda x:'Max' in x and 'project' in x and 'checkpoint' not in x, grades.columns))]
    project_grades = grades[list(filter(lambda x:'project' in x and 'checkpoint' not in x and '-' not in x, grades.columns))]
    
    for n in range(1, project_count+1):
        total_score = project_grades[list(filter(lambda x: str(n) in x, project_grades.columns))].sum(axis=1)
        max_score = max_grades[list(filter(lambda x: str(n) in x, max_grades.columns))].sum(axis=1)
        overall_grades.append(total_score / max_score)
    
    return pd.Series(sum(overall_grades) / project_count)

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------
def time_converter(time):
    h = int(time[0])
    m = int(time[1])
    s = int(time[2])
    return h*3600 + m*60 + s


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    late_count = []
    lab_count = len(get_assignment_names(grades)['lab'])
    all_labs = grades[list(filter(lambda x: 'lab' in x and 'Lateness' in x, grades.columns))]
    
    for n in range(1, lab_count+1):
        late_times = all_labs[list(filter(lambda x: str(n) in x, all_labs.columns))[0]].str.split(":")
        seconds_late = late_times.apply(time_converter)
        late_count.append(len(seconds_late[seconds_late > 0][seconds_late <= 10800]))
    return pd.Series(late_count,get_assignment_names(grades)['lab'])

# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    reformatted = col.str.split(":")
    seconds_late = reformatted.apply(time_converter)
    scores = [1 if time <= 10800 else 0.9 if time <=604800 else 0.7 if time <= 1209600 else 0.4 for time in seconds_late]
    return pd.Series(scores)

# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    
    all_labs = grades[list(filter(lambda x: 'lab' in x and '-'not in x, grades.columns))]
    lab_cols = all_labs.columns
    max_points = grades[list(filter(lambda x: 'lab' in x and 'Max'in x, grades.columns))]
    lateness = grades[list(filter(lambda x: 'lab' in x and 'Lateness' in x, grades.columns))]
    
    late_dict = {}
    for col in lateness.columns:
        late_dict[col] = lateness_penalty(lateness[col])
    lateness_df = pd.DataFrame(late_dict)
    
    penalty = pd.DataFrame(all_labs.values / max_points.values * lateness_df.values)
    
    final_grades = pd.DataFrame(penalty)
    final_grades.columns = lab_cols
    return final_grades

# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def find_highest(row):
    smallest = row.min()
    return (row.sum() - smallest) / (row.shape[0] - 1)

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
     
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    processed = processed.fillna(0)
    highest_scores = processed.apply(find_highest, axis=1)
    return highest_scores

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------
def points_helper(grades, col):
    
    col_all = grades[list(filter(lambda x: col in x and '-' not in x, grades.columns))]
    col_max = grades[list(filter(lambda x: col in x and 'Max' in x, grades.columns))]
    col_grades = pd.DataFrame(col_all.values / col_max.values)
    col_avg_grades = col_grades.mean(axis=1)#.reset_index()
    
    return col_avg_grades

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades = grades.fillna(0)
    
    lab_grades = lab_total(process_labs(grades))
    lab_perc = .2 * lab_grades
    
    project_grades = projects_total(grades)
    project_perc = .3 * project_grades
    
    cp_avg_grades = points_helper(grades, 'checkpoint')
    checkpoint_perc = cp_avg_grades * .025
    
    disc_avg_grades = points_helper(grades, 'discussion')
    discussion_perc = disc_avg_grades * .025
    
    mexam_avg_grades = points_helper(grades, 'Midterm')
    mexam_perc = mexam_avg_grades * .15
    
    fexam_avg_grades = points_helper(grades, 'Final')
    fexam_perc = fexam_avg_grades * .3
    
    finalgrades = lab_perc + project_perc + checkpoint_perc + discussion_perc + mexam_perc + fexam_perc
    
    return finalgrades

def letter_helper(value):
    grade_dict = {90: "A",80: "B",70: "C",60: "D",0: "F"}
    for key, letter in grade_dict.items():
        if value >= key:
            return letter

def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    
    total = (total * 100).astype(int)
    letter_grades = total.map(letter_helper)
    
    return letter_grades


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    num_grades = total_points(grades)
    let_grades = final_grades(num_grades)
    #print(num_grades.shape[0])
    #print(let_grades.value_counts().values.sum())
    return pd.Series(let_grades.value_counts().values / num_grades.shape[0], index=let_grades.value_counts().index)

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    
    totals = total_points(grades)
    grades['Grade'] = totals
    cleaned = grades[['PID', 'Level', 'Grade']]
    grouped_means = cleaned.groupby('Level').mean().reset_index()
    observed = grouped_means.loc[grouped_means.Level == 'SR'].values[0][1]
    
    seniors = grades[grades['Level'] == 'SR']
    
    cat_distr = cleaned['Grade'].value_counts(normalize=True)
    samples = np.random.choice(cat_distr.index,p=cat_distr,size=(N,int(seniors.shape[0])))
    
    averages = samples.mean(axis=1)    
    
    pval = np.count_nonzero(averages <= observed) / N
    return pval

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------
def noisy_helper_dcp(grades, col):
    names = get_assignment_names(grades)[str(col)]
    col_all = grades[names]
    col_list = list(filter(lambda x: col in x and 'Max' in x, grades.columns))
    col_max = grades[col_list]
    grade_dict = {}
    
    for col in range(len(names)):
        processed = pd.DataFrame(col_all[names[col]] / col_max[col_list[col]])
        processed += np.random.normal(0, 0.02, size=(processed.shape[0],processed.shape[1]))
        processed = np.clip(processed.iloc[:,0],0,1)
        grade_dict[names[col]] = processed
    points_df = pd.DataFrame(grade_dict)
    points_df = points_df.fillna(0)
    averages = points_df.mean(axis=1)
    
    return averages

def noisy_helper_exam(grades, exam):
    capitalized_exam = exam.capitalize()
    scores = grades[capitalized_exam]
    maxes = list(filter(lambda x: capitalized_exam in x and 'Max' in x, grades.columns))
    max_points = grades[maxes[0]]
    noisy_grades = pd.DataFrame(scores / max_points)
    noisy_grades += np.random.normal(0, 0.02, size=(noisy_grades.shape[0], noisy_grades.shape[1]))
    noisy_grades = np.clip(noisy_grades.iloc[:,0],0,1)
  
    return noisy_grades

def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades.fillna(0, inplace=True)
    
    proc_labs = process_labs(grades)
    proc_labs += np.random.normal(0, 0.02, size=(proc_labs.shape[0], proc_labs.shape[1]))
    noisy_lab_grades = lab_total(np.clip(proc_labs,0,1)) * 20
    
    total_percs = []
    assignments = get_assignment_names(grades)
    project_count = len(assignments['project'])
    proj = grades[list(filter(lambda x: 'project' in x and '-' not in x and 'checkpoint' not in x, grades.columns))]
    proj = proj.fillna(0)
    proj_max = grades[list(filter(lambda x: 'Max' in x and 'project' in x and 'checkpoint' not in x, grades.columns))]    
    for num in range(1, project_count+1):
        final_proj_scores = proj[list(filter(lambda x: str(num) in x, proj.columns))].sum(axis=1)
        max_proj_points = proj_max[list(filter(lambda x: str(num) in x, proj_max.columns))].sum(axis=1)
        final_proj_grades = pd.DataFrame(final_proj_scores / max_proj_points)
        
        final_proj_grades += np.random.normal(0, 0.02, size=(final_proj_grades.shape[0], final_proj_grades.shape[1]))
        noisy_proj_grades = np.clip(final_proj_grades.iloc[:,0],0,1) * 30
        total_percs.append(noisy_proj_grades)
    noisy_proj_grades = sum(total_percs) / project_count 
    
    noisy_cp_grades = noisy_helper_dcp(grades, 'checkpoint')
    
    noisy_disc_grades = noisy_helper_dcp(grades, 'disc')
    
    noisy_mexam_grades = noisy_helper_exam(grades, 'midterm') * 15
    
    noisy_fexam_grades = noisy_helper_exam(grades, 'final') * 30
    
    total = noisy_lab_grades + noisy_proj_grades + noisy_disc_grades + noisy_cp_grades + noisy_mexam_grades + noisy_fexam_grades
    
    return total / 100

# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """

    return [0.007, 84.673, [78.00,86.17], .065, [True,False]]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
