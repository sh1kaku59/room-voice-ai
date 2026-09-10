import tarfile

with tarfile.open("C:/temp/webrtc_noise_gain-1.3.0.tar.gz", "r:gz") as tar:
    tar.extractall("C:/temp/")

setup_file = "C:/temp/webrtc_noise_gain-1.3.0/setup.py"
with open(setup_file, encoding="utf-8") as f:
    content = f.read()

# 1. Replace POSIX rw_lock with Windows rw_lock
content = content.replace('"synchronization/rw_lock_posix.cc"', '"synchronization/rw_lock_win.cc"')

# 2. Fix broken quote literals
content = content.replace('"\\\'', '"').replace('\\\'"', '"')
content = content.replace('"\'', '"').replace('\'"', '"')

# 3. Add the required Abseil C++ files on Windows (including memmatch.cc)
old_win = 'system == "windows":\n    system_cflags += ['
new_win = '''system == "windows":
    absl_sources = [
        "absl/types/bad_optional_access.cc",
        "absl/base/internal/throw_delegate.cc",
        "absl/strings/string_view.cc",
        "absl/strings/internal/memmatch.cc",
    ]
    system_cflags += ['''
content = content.replace(old_win, new_win)

with open(setup_file, "w", encoding="utf-8") as f:
    f.write(content)
print("Added memmatch.cc to webrtc_noise_gain on Windows!")
