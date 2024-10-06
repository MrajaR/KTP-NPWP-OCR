ktp_patterns = {
    'nama': r'Nama\s+:\s+([^\n]+)',  # Capture everything after "Nama : " until the end of the line
    'tempat_lahir': r'Tempat/Tgl\s+Lahir\s+:\s+([^\n,]+)',  # Capture place of birth
    'tgl_lahir': r'Tempat/Tgl\s+Lahir\s+:\s+[^\n,]+,\s+([\d\-]+)',  # Capture date of birth
    'alamat': r'Alamat\s+:\s+([^\n]+)',  # Capture everything after "Alamat : " until the end of the line
    'agama': r'Agama\s+:\s+(\w+)',  # Capture the word after "Agama :"
    'pernikahan': r'Status\s+Perkawinan\s+:\s+([^\n]+)',  # Capture everything after "Status Perkawinan : " until the end of the line
    'pekerjaan': r'Pekerjaan\s+:\s+([^\n]+)',  # Capture everything after "Pekerjaan : " until the end of the line
    'golongan_darah': r'Gol\.\s+Darah\s+:\s*(\w+|\n)',  # Capture the word after "Gol. Darah :"
    'jenis_kelamin': r'Jenis\s+Kelamin\s+:\s+([^\n]+)',  # Capture everything after "Jenis Kelamin : " until the end of the line
    'kecamatan': r'Kecamatan\s+:\s+([^\n]+)',  # Capture everything after "Kecamatan : " until the end of the line
    'rtrw': r'RT/RW\s+:\s+(\d+/\d+)',  # Capture the "RT/RW" format like "###/###"
    'nik': r'NIK\s+:\s+(\d+)',  # Capture the NIK (sequence of digits)
    'kewarganegaraan': r'Kewarganegaraan\s+:\s+([^\n]+)',  # Capture the word after "Kewarganegaraan :"
    'provinsi': r'PROVINSI\s+:\s+([^\n]+)',  # Capture everything after "PROVINSI " until the end of the line
    'kota_kabupaten': r'KOTA/KABUPATEN\s+:\s+([^\n]+)',  # Capture everything after "KOTA/KABUPATEN : " until the end of the line
    'kel_desa': r'Kel/Desa\s+:\s+([^\n]+)',  # Capture everything after "Kel/Desa : " until the end of the line
    'masa_berlaku': r'Berlaku\s+Hingga\s+:\s+([^\n]+)'  # Capture everything after "Berlaku Hingga : " until the end of the line
}

npwp_patterns = {        
    'npwpId': r'NPWP\s*:\s*([\d\.\-]+)',  # Extract NPWP ID (includes digits, dots, and dashes)
    'nama': r'NPWP\s*:\s*[\d\.\-]+\s+([^\n]+)\s+NIK\s*:',  # Extract name between NPWP and NIK
    'nik': r'NIK\s*:\s*(\d{16})',  # Extract NIK (16 digits)
    'alamat': r'\d{16}\s*((?:.|\s)*?)\s*KPP',  # Extract address (non-greedy until "KPP")
    'kpp': r'KPP\s+([^\n]+)\s+Terdaftar\s*:',  # Extract KPP office between "KPP" and "Terdaftar:"
    'tanggal_terdaftar': r'Terdaftar\s*:\s*(\d+\s\w+\s\d{4})'  # Extract registration date after "Terdaftar:"
}




