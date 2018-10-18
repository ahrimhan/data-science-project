# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011

# Update the 2012 key in the boy_names dictionary
boy_names[2012].update([(1, 'Casey'), (2, 'Aiden')])

# Loop over the boy_names dictionary 
for year in boy_names:
    # Loop over and sort the data for each year by descending rank
    #print(sorted(boy_names[year], reverse=True))
    for rank in sorted(boy_names[year], reverse=True)[:1]:
        # Check that you have a rank
        if not rank:
            print(year, 'No Data Available')
        # Safely print the year and the least popular name or 'Not Available'
        print(year, boy_names[year].get(rank, 'Not Available'))
