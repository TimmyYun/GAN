import youtube_dl
filename='ize_eiCFEg0'
ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
	ydl.download([f'https://www.youtube.com/watch?v={filename}'])
