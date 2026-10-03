from PIL import Image,ImageDraw
from pathlib import Path
p=Path(r'results/paraview_bauxitemed_trace/figures')
names=['bauxitemed_final_pit_isometric_blocks.png','bauxitemed_trace_stages.png','bauxitemed_final_pit_by_partition.png','bauxitemed_repair_cascade.png','bauxitemed_initial_violations.png','bauxitemed_boundary_changes.png']
thumbs=[]
for name in names:
 im=Image.open(p/name).convert('RGB'); im.thumbnail((700,460))
 cv=Image.new('RGB',(720,500),'white'); cv.paste(im,((720-im.width)//2,0)); ImageDraw.Draw(cv).text((10,478),name,fill='black'); thumbs.append(cv)
out=Image.new('RGB',(1440,1500),'#ddd')
for i,im in enumerate(thumbs): out.paste(im,((i%2)*720,(i//2)*500))
out.save(p/'figure_contact_sheet.png')
print('updated contact sheet')
