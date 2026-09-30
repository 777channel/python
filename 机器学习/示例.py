def is_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    return False
if __name__ == "_main_":
    print("2004年：is_year(2004)")
    print("2000年：is_year(2000)")
    print("1980年：is_year(1980)")

