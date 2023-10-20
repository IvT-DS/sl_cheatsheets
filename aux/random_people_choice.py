from numpy.random import choice, shuffle
from typing import List

def random_people_choice(people: list, teams_names: list) -> dict:
    teams = {}
    p = len(people)
    t = len(teams_names)

    for team_n in teams_names:
        team = []
        for i in range(int(p/t)):
            name = choice(people)
            people.remove(name)
            team.append(name)
        teams[team_n] = team
    
    if len(people)>0:
        for name in people:
            try:
                team = choice(list(teams.keys()))
                people.remove(name)
                teams[team].append(name)
            except Exception as e:
                pass
    # st.write(teams)
    return teams

# def get_teams(peoples: list, teams: int):
#     shuffle(peoples)
#     pairs = dict()
#     for team_index, i in enumerate(range(0, len(peoples), len(teams))):
#         pairs[teams[team_index]] = peoples[i:i + len(teams)]
#     return pairs

def get_teams(students: List[str], teams: List[str]) -> dict:
    
    #Shuffle list of students
    shuffle(students)
    
    #Create groups
    all_groups = []
    for index in range(len(teams)):
        group = students[index::len(teams)]
        all_groups.append(group)
    
    #Format and display groups
    # for team, group in zip(teams, all_groups):
    #     print(f"✨Group {team}✨: {' / '.join(group)}\n")

    return {team : group for team, group in zip(teams, all_groups)}
    