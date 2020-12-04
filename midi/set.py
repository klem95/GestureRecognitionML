import live
print('set.py: init ableton set...')
set = live.Set()
print('set.py: scanning set...')
set.scan(scan_clip_names = True, scan_devices = True)
print('set.py: done')
set.caching = True