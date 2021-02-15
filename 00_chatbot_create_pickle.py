#answer.txt,question.txtから使用されている文字のpickleを生成する
import glob
import codecs
import re
import pickle

files = glob.glob("./text/*")
text = ""
for file in files:
    with open(file, mode="r", encoding="shift_jis") as f:  # ファイルの読み込み
    #with codecs.open(file, 'r', 'shift_jis', 'ignore') as f:
        text_novel = f.read()
        #text_novel = re.sub("[ 　\n「」『』（）｜※＊…]", "", text_novel)  # 全角半角スペース、改行、その他記号の削除
        text += text_novel

#print("文字数:", len(text))
#print(text)

seperator = "@"
sentence_list = text.split(seperator) 
sentence_list.pop() 
sentence_list = [x+seperator for x in sentence_list]

# for sentence in sentence_list:
#     print(sentence)

#print(set(text))  # set()で文字の重複をなくす
#print(text)

chars = ""
for char in text:  # ひらがな、カタカナ以外でコーパスに使われている文字を追加
    if (char not in chars) and char != "@":
        chars += char
chars += "\t\n"

chars_list = sorted(list(set(chars)))  # 文字列をリストに変換してソートする
print(chars_list)

with open("chars.pickle", mode="wb") as f:  # pickleで保存
    pickle.dump(chars_list, f)