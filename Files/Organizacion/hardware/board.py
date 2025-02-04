from brainflow import BoardShim, BrainFlowInputParams, LogLevels, BoardIds


# Función para manejar datos sintéticos u reales (OpenBCI)
def setup_board(is_synthetic):
    params = BrainFlowInputParams()
    
    if is_synthetic:
        board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        print("Using synthetic values...")
    else:
        open_bci_com=variables['test_parms']['open_bci_com']
        params.serial_port = 'COM' + open_bci_com
        board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        print(f"BCI connected on COM{open_bci_com}.")
    
    board.prepare_session()
    timestamp_channel = board.get_timestamp_channel(BoardIds.CYTON_BOARD.value if not is_synthetic else BoardIds.SYNTHETIC_BOARD.value)
    acc_channel = board.get_accel_channels(BoardIds.CYTON_BOARD.value if not is_synthetic else BoardIds.SYNTHETIC_BOARD.value)
    
    return board, timestamp_channel, acc_channel
