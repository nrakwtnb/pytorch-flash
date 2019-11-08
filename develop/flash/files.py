import os
import glob

def prepare_target_dir(target_dir, assert_on=True):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        return True
    if any(map(os.path.isfile, glob.glob(os.path.join(target_dir, '**'), recursive=True))):
        if assert_on:
            assert False, 'already exists !'
        else:
            i = 0
            while i < 3:
                print(f"{target_dir} is aleready exists: Do you continue ? (y/n)")
                user_reponse = input().lower()
                if user_reponse == 'y':
                    break
                elif user_reponse == 'n':
                    import sys
                    sys.exit(0)
                i += 1
            return False
    else:
        return True

