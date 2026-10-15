database checking for ai chat commands 
Do this in Command Prompt while your Flask server can be stopped temporarily:

cd C:\SANU\NeuroGuardian
thenn type this 
python -c "import sqlite3; db=sqlite3.connect('database/neuroguardian.db'); c=db.cursor(); print(c.execute('SELECT id,user_email,role,message,created_at FROM chat_messages ORDER BY id').fetchall()); db.close()"

cmd for checking db of migraine analysis
python -c "import sqlite3; db=sqlite3.connect('database/neuroguardian.db'); c=db.cursor(); print(c.execute('SELECT id,user_email,score,title,priority,pain,location,symptoms,triggers,emergency,timestamp,created_at FROM migraine_diary ORDER BY id').fetchall()); db.close()"