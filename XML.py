#!/usr/bin/python3

import xml.sax
from os import listdir
import os
import time



class MovieHandler( xml.sax.ContentHandler ):
   def __init__(self):
      self.CurrentData = ""
      self.user = ""
      self.username = ""
      self.body = ""
      self.content=""

   # Call when an element starts
   def startElement(self, tag, attributes):
      self.CurrentData = tag
      if tag == "posts":
         print ("*****Content*****")

   # Call when an elements ends
   def endElement(self, tag):
      """if self.CurrentData == "user":
         self.content+= "User:"+ self.user+"\n"
         #file_handle.write(self.user)
      elif self.CurrentData == "username":
         self.content+="Username:"+ self.username+"\n"
         #file_handle.write(self.user)"""
      if self.CurrentData == "body":
         self.content+="Body:"+ self.body+"\n"
         write_to.write(self.body+"\n")
         print(self.body+"\n")
      self.CurrentData = ""

   # Call when a character is read
   def characters(self, content):
      if self.CurrentData == "user":
         self.user = content
      elif self.CurrentData == "username":
         self.username = content
      elif self.CurrentData == "body":
         self.body = content
  
   


# override the default ContextHandler

"""folder information"""

"""remove comments for this part to parse xml
# create an XMLReader
parser = xml.sax.make_parser()
# turn off namepsaces
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

#fetches renamed files from the xmlpacket folder
files = sorted([f for f in listdir('./xmlpacket2')])

#textpacket location
location = "./textpacket2/"

#xmlpacket location
xml_location = "./xmlpacket2/"

#the folder number e.g. xmlpacket1 is foldernumber 0 ...
folder_num=str(1)

num=-1
for f in files:
   num+=1
   Handler = MovieHandler()
   parser.setContentHandler( Handler )
   #read files
   file_handle = open(xml_location+f, "r")
   try:
      write_to = open(location+folder_num+"file"+str(num)+".txt", "w")
      parser.parse(xml_location+f)
   except xml.sax._exceptions.SAXParseException:
      print("Empty")
   file_handle.close()"""

def change_filename():
   """changes xml file names e.g. 0file1.xml"""
   #change location for xml packet location and to get certain files to keep file order
   first_digit_packets = sorted([f for f in listdir('./xmlpacket2') if len(f)<14 ])
   second_digit_packets = sorted([f for f in listdir('./xmlpacket2') if len(f)==16 ])
   #ensure packets exist 
   print("total length",len(first_digit_packets)+len(second_digit_packets))
   #index is the folder number - change this
   index=0
   file_num=0
   for f in second_digit_packets:
      print("file",f)
      file_num+=1
      com = str(index)+"file"+str(file_num)+".xml"
      location = "./xmlpacket2/"
      print(location)
      os.rename(location+f,location+com)
