import cv2
import time
import numpy as np
import hand_tracking_module as htm
import cvzone
import random


class Button:
    def __init__(self, pos, text, size=None, font_scale=1.0):
        if size is None:
            size = [100, 100]
        self.pos = pos
        self.size = size
        self.text = text
        self.font_scale = font_scale


def draw_all(img, button_list):
    img_new = np.zeros_like(img, np.uint8)
    for button in button_list:
        x, y = button.pos
        cvzone.cornerRect(img_new, (button.pos[0], button.pos[1], button.size[0] + 50, button.size[1]),
                          20, rt=0)
        cv2.rectangle(img_new, button.pos, (x + button.size[0] + 50, y + button.size[1]),
                      (255, 0, 255), cv2.FILLED)
        cv2.putText(img_new, button.text, (x + 40, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    out = img.copy()
    alpha = 0.5
    mask = img_new.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, img_new, 1 - alpha, 0)[mask]
    return out


def check_board(board):
    for row in board:
        if all(x == "X" for x in row):
            return "X"
        elif all(x == "O" for x in row):
            return "O"
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != "":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "":
        return board[0][2]
    return None


def update_board(board_list, board):
    for i in range(9):
        x, y = i // 3, i % 3
        if board_list[i].text == "X":
            board[x][y] = "X"
    while True:
        free_spots = any("" in row for row in board)
        if not free_spots:
            break
        x, y = random.randint(0, 2), random.randint(0, 2)
        if board[x][y] == "":
            board[x][y] = "O"
            break
    for i in range(9):
        x, y = i // 3, i % 3
        if board[x][y] == "O":
            board_list[i].text = "O"


def main():
    p_time = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    detector = htm.HandDetector()
    keys = [["START"]]
    board = [["", "", ""],
             ["", "", ""],
             ["", "", ""]]
    button_list = []
    board_list = []
    for x, key in enumerate(keys[0]):
        button_list.append(Button([800 * x + 50, 100], key))
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        img = draw_all(img, button_list)
        if lm_list:
            for button in button_list:
                x, y = button.pos
                w, h = button.size
                if x < lm_list[8][1] < x + w and y < lm_list[8][2] < y + h:
                    x, y = button.pos
                    cv2.rectangle(img, button.pos, (x + button.size[0] + 50, y + button.size[1]),
                                  (255, 0, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 40, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                    length = detector.find_distance_between_fingers(img, p1=8, p2=12, pp1=1, pp2=1, pp3=2, pp4=2,
                                                                    draw=False)
                    if length < 50:
                        cv2.rectangle(img, button.pos, (x + button.size[0] + 50, y + button.size[1]),
                                      (255, 13, 13), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 40, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                        time.sleep(0.15)
                        button_list.clear()
                        for i in range(len(board)):
                            for j, btn in enumerate(board[i]):
                                board_list.append(Button([370 * j + 400, 320 * i + 50], btn, size=[300, 300], font_scale=10))
            for btn in board_list:
                x, y = btn.pos
                w, h = btn.size
                if x < lm_list[8][1] < x + w and y < lm_list[8][2] < y + h:
                    x, y = btn.pos
                    cv2.rectangle(img, btn.pos, (x + btn.size[0] + 50, y + btn.size[1]),
                                  (255, 0, 255), cv2.FILLED)
                    length = detector.find_distance_between_fingers(img, p1=8, p2=12, pp1=1, pp2=1, pp3=2, pp4=2,
                                                                    draw=False)
                    if length < 50:
                        time.sleep(0.30)
                        btn_rect = (btn.pos[0], btn.pos[1], btn.size[0] + 50, btn.size[1])
                        cv2.rectangle(img, btn_rect, (255, 13, 13), cv2.FILLED)
                        if btn.text != "O":
                            btn.text = "X"
                        update_board(board_list, board)
        result = check_board(board)
        if result is not None:
            board_list.clear()
            text = "The game winner is " + result
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (255, 0, 255)
            cv2.putText(img, text, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        img = draw_all(img, board_list)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255))
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
