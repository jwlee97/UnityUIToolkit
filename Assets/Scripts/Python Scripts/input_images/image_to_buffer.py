import base64

def img_to_buff(in_file):
    encoded = base64.b64encode(open(in_file, "rb").read())   

    out_file_name = in_file.replace('.PNG', '_buff.log')
    out_file = open(out_file_name, "wb")
    out_file.write(encoded)
    out_file.close()

def main():
    img_to_buff("office.PNG")
    img_to_buff("classroom.PNG")
    img_to_buff("lab.PNG")

if __name__ == "__main__":
    main()