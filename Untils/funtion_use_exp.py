season = ['spring', 'summer', 'fall', 'winter']
print(list(enumerate(season)))
season_list = list(enumerate(season))
season_list[0]
#list(enumerate)[1]
input()
list(enumerate(season))
list(enumerate(season,start = 1))

# def enumerate(sequence, start=0):
#     n = start
#     for elem in sequence:
#         yield n, elem
#         n +=1