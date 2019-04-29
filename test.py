from embedding.dhpe import DHPE

file = 'data/fb.csv'

embedding = DHPE(file, type='undirected')
embedding.embed()