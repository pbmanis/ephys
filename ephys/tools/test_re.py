import re
""" Parse S0C0, S0C00, S00C00 to 
the slice # and cell # 
"""
re_parse = re.compile("([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")

# test cases:
tests = ["S0C0", "S0C00", "S00C00", "S001C001"]
for test in tests:
    print(test, 'all groups => ',re_parse.match(test).groups())
    print(test, 'group 1 => ',re_parse.match(test).group(1))
    print(test, 'group 2 => ',re_parse.match(test).group(2), int(re_parse.match(test).group(2)))
    print(test, 'group 3 => ',re_parse.match(test).group(3))
    print(test, 'group 4 => ',re_parse.match(test).group(4), int(re_parse.match(test).group(4)))
    print("-"*80)
