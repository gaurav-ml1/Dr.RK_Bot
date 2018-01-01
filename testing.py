    import requests
    data = requests.get(url='https://raw.githubusercontent.com/dwyl/english-words/master/words.txt')
    mfile = open('english_dict.txt', 'w')
    mfile.write(str(data.text))