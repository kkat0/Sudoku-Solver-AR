from keras.models import load_model
import numpy as np
import cv2

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
MIN_MATCH_COUNT = 15
TEMPLATE_SIZE = 350

def main():
    brisk = cv2.BRISK.create()
    template = None
    tkps, tdes = None, None
    overlay_img = None

    index_params = {
        'algorithm': FLANN_INDEX_LSH,
        'table_number': 6,
        'key_size': 12,
        'multi_probe_level': 1
    }
    search_params = {
        # 'checks': 30
    }

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(1)
    if cap == None:
        print("Failed to open camera")
        exit(1)
    fheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = 480
    width = int(height * fwidth / fheight)

    cell_clf = load_model('cell_recognizer.h5')

    while True:
        ret, frame = cap.read()
        src = cv2.resize(frame, (width, height))
        dst = src.copy()

        binary = None
        rect = None

        if template is None:
            binary = preprocess(src)
            rect = find_rect(binary)
            if rect is not None:
                dst = cv2.polylines(dst, [rect], True, (0, 0, 255), 5)
        else:
            skps, sdes = brisk.detectAndCompute(src, None)
            
            if 2 <= len(tkps) and 2 <= len(skps):
                matches = flann.knnMatch(tdes, sdes, k=2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.6]

                # out = np.zeros((100, 100))
                # out = cv2.drawMatches(template, tkps, src, skps, good, out, flags=2)
                # cv2.imshow('match', out)

                if len(good) > MIN_MATCH_COUNT:
                    tpts = np.float32([tkps[m.queryIdx].pt for m in good])
                    spts = np.float32([skps[m.trainIdx].pt for m in good])
                    M, mask = cv2.findHomography(tpts, spts, cv2.RANSAC, 3.0)

                    pts = np.float32([[0,0],[0, TEMPLATE_SIZE-1],[TEMPLATE_SIZE-1,TEMPLATE_SIZE-1],[TEMPLATE_SIZE-1,0]]).reshape(-1,1,2)
                    dst_pts = cv2.perspectiveTransform(pts, M)
                    dst = cv2.polylines(dst, [np.int32(dst_pts)], True, (0, 255, 255), 3, cv2.LINE_AA)

                    # out = np.zeros((100, 100))
                    # out = cv2.drawMatches(template, tkps, dst, skps, good, out, flags=2)
                    # cv2.imshow('match', out)

                    dst = overlay(dst, overlay_img, M)

        cv2.imshow('src', src)
        cv2.imshow('dst', dst)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

        if key == 32 and rect is not None:
            template = extract(src, rect)
            template_bin = extract(binary, rect)
            tkps, tdes = brisk.detectAndCompute(template, None)
            cells, mask = recognize(template_bin, cell_clf)
            solved, filled = solve_sudoku(cells)
            if solved:
                overlay_img = gen_overlayimg(filled, mask)
            else:
                overlay_img = gen_overlayimg(cells, mask)

        if key == ord('c'):
            template = None
            tkps, tdes = None, None
        
        if key == ord('s'):
            cv2.imwrite("src.png", src)
            cv2.imwrite("dst.png", dst)

    cap.release()
    cv2.destroyAllWindows()


def preprocess(src):
    tmp = cv2.bilateralFilter(src, 5, 7, 7)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        tmp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 15)

    # cv2.imshow("binary", binary)

    return binary


def find_rect(src):
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = list(filter(lambda x: len(x) == 4, [cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True)*0.05, True) for cnt in contours]))

    if len(rects) == 0:
        return None

    rect = sorted(rects, key=lambda x: cv2.contourArea(x),
                reverse=True)[0].reshape(4, 2)
    area = cv2.contourArea(rect)

    if area < 180 ** 2:
        return None

    return rect


def extract(src, rect):
    idx = 0
    min_dist = float('inf')

    for i, p in enumerate(rect):
        dist = p[0] + p[1]

        if dist < min_dist:
            min_dist = dist
            idx = i

    p_src = np.ndarray(rect.shape, dtype=np.float32)
    p_dst = np.array([
        [0, 0],
        [0, TEMPLATE_SIZE],
        [TEMPLATE_SIZE, TEMPLATE_SIZE],
        [TEMPLATE_SIZE, 0],
    ], dtype=np.float32)

    for i in range(4):
        p_src[i] = rect[(idx + i) % 4]

    M = cv2.getPerspectiveTransform(p_src, p_dst)
    ret = cv2.warpPerspective(src, M, (TEMPLATE_SIZE, TEMPLATE_SIZE))

    return ret


def recognize(board, clf):
    board = cv2.resize(board, (252, 252))
    cells = np.ndarray((81, 28, 28, 1))

    for i in range(9):
        for j in range(9):
            cells[9 * i + j] = board[28*i:28 *
                                     (i+1), 28*j:28*(j+1)].reshape(28, 28, 1)

    pre = clf.predict(cells.astype(np.float32) / 255,
                      verbose=1).argmax(axis=1).reshape(9, 9)

    return pre, pre != 0


def gen_overlayimg(board, mask):
    board_img = None

    font = cv2.FONT_HERSHEY_PLAIN

    if board is None:
        board_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(board_img, 'Can not solve.', (10, 50),
                    font, 3, (0, 0, 255), 5, cv2.LINE_AA)
    else:
        board_img = np.ones((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(board_img, (0, 0), (500, 500), (255, 255, 255), -1)
        ls = np.linspace(0, 500, 10).astype(np.int)

        for i in range(10):
            board_img = cv2.line(
                board_img, (ls[i], 0), (ls[i], 500), (0, 0, 0), 3)
            board_img = cv2.line(
                board_img, (0, ls[i]), (500, ls[i]), (0, 0, 0), 3)

        for i in range(9):
            for j in range(9):
                tx = ls[j] + 8
                ty = ls[i] + 50
                if board[i, j] != 0:
                    if mask[i, j]:
                        cv2.putText(board_img, str(
                            board[i, j]), (tx, ty), font, 4, (255, 255, 0), 5, cv2.LINE_AA)
                    else:
                        cv2.putText(board_img, str(
                            board[i, j]), (tx, ty), font, 4, (255, 0, 0), 5, cv2.LINE_AA)

    return board_img


def overlay(base_img, overlay_img, M):
    overlay_img = cv2.resize(overlay_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))

    warped = cv2.warpPerspective(
        overlay_img, M, (base_img.shape[1], base_img.shape[0])).astype(np.uint8)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1 = cv2.bitwise_and(warped, warped, mask=mask)
    img2 = cv2.bitwise_and(base_img, base_img, mask=mask_inv)
    dst = cv2.add(img1, img2)

    return dst

def solve_sudoku(board):
    return solve_dfs(board, 0)


def solve_dfs(board, i):
    if i == 81:
        return True, board.copy()

    if board[i // 9, i % 9] != 0:
        return solve_dfs(board, i + 1)

    bx = (i % 9) // 3
    by = i // 27

    for n in range(1, 10):
        ok = True
        for j in range(9):
            if board[i // 9, j] == n or board[j, i % 9] == n or board[3 * by + (j // 3), 3 * bx + (j % 3)] == n:
                ok = False

        if not ok:
            continue

        board[i // 9, i % 9] = n
        ret = solve_dfs(board, i + 1)
        board[i // 9, i % 9] = 0

        if ret[0] == True:
            return ret

    return False, None


if __name__ == '__main__':
    main()
