# T-Shape MAPPO — Fixed Easy

Baseline kecil untuk belajar cooperative transport dengan MuJoCo dan shared-policy MAPPO.
Proyek saat ini hanya menjalankan **satu skenario Easy tetap dengan empat agen**.
Posisi awal objek, agen, target, dan orientasi selalu sama pada setiap episode.

Randomisasi environment, curriculum, arena penghalang, matriks eksperimen, serta
kompatibilitas checkpoint eksperimen lama sudah dihapus.

## Skenario

| Komponen | Konfigurasi awal |
| --- | --- |
| Ruang arena | 5,8 x 5,8 m, tanpa rintangan internal |
| Objek | Dua palang asimetris dan satu batang, satu rigid body |
| Dimensi objek | Panjang 1,2 m; palang besar 0,6 m; palang kecil 0,3 m |
| Massa objek | 2 kg |
| Posisi / yaw objek | (-1,5, -1,5), 0 rad — kiri bawah |
| Agen | Empat silinder, radius 0,03 m, massa 0,25 kg per agen |
| Posisi agen | Dua di bawah palang besar, dua di kiri batang, dekat objek |
| Pusat target | (1,5, 1,5) — kanan atas |
| Ukuran penuh target | 1,5 x 1,3 m |
| Horizon | 500 keputusan, setara 25 detik simulasi |
| Kontrol | Dua gaya kontinu per agen, maksimal +/-3 N per sumbu |

Objek mulai di kiri bawah dan target tetap di kanan atas. Keempat agen mulai dekat
objek, dengan gap permukaan 4 cm; sepasang dapat mendorong ke atas dan sepasang ke
kanan. Gerak X/Y dan rotasi objek mengikuti fisika.

Gerak planar memakai joint X/Y dan yaw objek dengan gravitasi nol. Lantai bersifat
visual. Resistansi objek berasal dari damping dan frictionloss joint, bukan beban
normal lantai. Parameter fisika berada di [arena.xml](assets/mujoco/arena.xml).

Setiap keputusan menahan aksi selama 50 substep x 0,001 s = 0,05 s, sehingga policy
berjalan pada 20 Hz. Reset selalu memulihkan layout dan kecepatan awal yang sama.
Seed mengendalikan jaringan, sampling aksi, dan pembaruan PPO; seed tidak mengubah layout.
Eksplorasi aksi saat training tetap digunakan.

## Instalasi

Gunakan Python 3.10 dan environment Python proyek:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
~~~

Pilih build PyTorch sesuai CPU/CUDA mesin. Periksa ketersediaan CUDA sebelum memakai
opsi device cuda. Video memerlukan FFmpeg pada PATH:

~~~bash
sudo apt install ffmpeg
~~~

Untuk hanya menjalankan demo fisika, cukup instal requirements.txt.

## Cek fisika terlebih dahulu

~~~bash
python scripts/demo.py
python scripts/demo.py --render
~~~

Demo menggunakan controller referensi untuk layout bawaan. Controller ini membaca
geometri secara langsung, tidak dipakai trainer, dan bukan policy yang dipelajari.
Gunanya memastikan bahwa kontak fisik dapat membawa seluruh objek ke target.

## Training dari awal

~~~bash
python scripts/train.py --device cuda --total-steps 100000 --run-dir results/easy_fixed
~~~

Gunakan device cpu untuk komputer tanpa CUDA. Run directory harus baru.
Bobot actor dan critic diinisialisasi dari awal.

Tanpa video otomatis:

~~~bash
python scripts/train.py --device cpu --total-steps 100000 --video-interval-steps 0 --run-dir results/easy_fixed_cpu
~~~

Konfigurasi statis berada di [environment.yaml](configs/environment.yaml), sedangkan
hyperparameter ada di [mappo.yaml](configs/mappo.yaml). Koordinat dapat diedit secara
manual; setelah dipilih, koordinat tersebut tetap pada seluruh episode.

Trainer menggunakan satu arena. MuJoCo berjalan pada CPU; jaringan dapat berjalan
pada CPU atau GPU.

## Evaluasi

~~~bash
python scripts/evaluate.py results/easy_fixed/checkpoints/best.pt --device auto --render
python scripts/evaluate.py results/easy_fixed/checkpoints/latest.pt --device auto --video-dir results/easy_fixed/videos/manual --json results/easy_fixed/manual_evaluation.json
~~~

Rekaman pertama disimpan sebagai policy.mp4 di video directory. Hanya episode
pertama direkam, termasuk ketika jumlah episode evaluasi lebih dari satu.

Evaluasi default memakai satu episode deterministik. Mengulang episode deterministik
pada layout tetap menghasilkan skenario yang sama; hasil tersebut bukan benchmark
generalisasi. Untuk melihat variasi akibat sampling aksi:

~~~bash
python scripts/evaluate.py results/easy_fixed/checkpoints/best.pt --stochastic --episodes 10
~~~

Script mencetak progres saat memuat policy, memulai episode, dan selama evaluasi.

## Resume training baru ini

~~~bash
python scripts/train.py --resume results/easy_fixed/checkpoints/latest.pt --device cuda --total-steps 200000 --run-dir results/easy_fixed_resume
~~~

Budget bersifat kumulatif: contoh tersebut melanjutkan sampai total 200.000 timestep.
Resume memulihkan model, optimizer, normalisasi dan RNG; simulasi mulai lagi pada
layout tetap. Output ditulis ke directory baru, dengan best yang dipilih dari evaluasi
pada run lanjutan tersebut.

Checkpoint menggunakan format baru khusus fixed Easy. Checkpoint versi eksperimen
sebelumnya tidak dimigrasikan atau dipakai sebagai inisialisasi.

## Observasi, state dan aksi

Actor menerima 34 fitur per agen: posisi/kecepatan diri, posisi objek dan target
relatif terhadap diri, sin/cos yaw objek, kecepatan objek, ukuran target, posisi
relatif serta kecepatan tiga agen lain, jarak ke batas luar arena, kontak dengan
objek, waktu tersisa dan progres hold sukses.

Observasi masih menggunakan informasi global. Tidak ada batas jangkauan sensor.
Posisi relatif memakai sumbu dunia; kecepatan agen lain masih absolut.

Critic menerima 33 fitur global: pose/kecepatan objek, target, seluruh posisi/kecepatan
agen, batas arena, waktu tersisa dan progres hold. Actor bersama berukuran
34 -> 128 -> 128 -> 2; critic 33 -> 256 -> 256 -> 1, dengan aktivasi Tanh.

Aksi adalah dua angka dalam [-1,1], dikalikan 3 N untuk gaya X dan Y masing-masing
agen. Actor menggunakan Gaussian dengan transformasi tanh saat training; evaluasi
deterministik menggunakan tanh dari mean. Semua agen memperoleh reward tim yang sama.

## Reward

Empat komponen yang diminta tetap digunakan, tanpa mode reward lama.
Dengan d = jarak pusat objek ke target, dan delta = selisih yaw terpendek:

~~~text
distance    = -d / (d + distance_scale)
orientation = -exp(-d / orientation_distance_scale) * abs(delta) / pi
progress    = previous_distance - d
success     = 1 hanya pada langkah terminal sukses

reward = 0.1*distance + 0.05*orientation + 2*progress + 20*success
~~~

Bobot dan kedua distance scale dapat dituning di environment.yaml. Goal yaw adalah
0 rad, dengan palang besar menuju +Y. Tidak ada penalti waktu tetap. Kegagalan
fisika mengganti reward menjadi -20 dan menonaktifkan empat komponen normal.

Distance/orientation adalah biaya yang mendekati nol ketika kondisi membaik.
Reward ini belum menjamin keberhasilan belajar; evaluasi policy tetap diperlukan.

## Sukses dan kegagalan

Sukses memerlukan seluruh sudut ketiga geometri objek masuk target, kecepatan linear
<=0,05 m/s, dan kecepatan yaw <=0,1 rad/s selama lima keputusan berturut-turut.
Pusat objek tidak harus tepat pada pusat area target.

Goal yaw merupakan preferensi reward; tidak ada toleransi sudut tambahan pada
syarat sukses. Timeout adalah truncation. Sukses atau kegagalan fisika adalah
termination. GAE melakukan bootstrap pada timeout dari state akhir sebelum reset.

Pemeriksaan setiap physics substep mendeteksi state non-finite, peringatan MuJoCo,
penetrasi >5 mm, linear speed >5 m/s, dan yaw speed >10 rad/s.

## Logging dan file hasil

Default: rollout 1.024 timestep tim, PPO lima epoch, gamma 0,99, GAE lambda 0,95,
learning rate actor/critic 0,0003. Satu timestep menerapkan aksi keempat agen bersama.

Evaluasi deterministik berlangsung setiap 10.000 timestep dan saat budget berakhir.
Checkpoint latest ditimpa setiap 10.000 timestep, pada video milestone, dan saat
training selesai normal. Best dipilih menurut success rate, lalu mean return.
Hanya best.pt dan latest.pt yang disimpan.

Video evaluasi 1920x1080 direkam setiap 100.000 timestep, di videos/step_<timestep>/policy.mp4.
Jadwal diperiksa sesudah update PPO, sehingga dapat sedikit melewati ambang jadwal.

File utama: manifest.json, environment.yaml, mappo.yaml, arena.xml, updates.jsonl,
episodes.jsonl, evaluations.jsonl, dan dua snapshot evaluasi latest/best.
Episodes.jsonl muncul setelah episode pertama selesai. Tidak ada file JSON per update.

~~~bash
tensorboard --logdir results --port 6006
~~~

Pantau success rate, return, distance, inside-goal steps, kontak per agen, serta
reward/distance_mean, reward/orientation_mean, reward/progress_mean dan reward/success_mean.
Loss kecil sendiri tidak menunjukkan bahwa transport berhasil.

## Struktur aktif

~~~text
assets/mujoco/arena.xml
configs/environment.yaml
configs/mappo.yaml
scripts/demo.py
scripts/train.py
scripts/evaluate.py
src/envs/          # Fisika dan Parallel API fixed Easy
src/agents/        # Shared actor dan centralized critic
src/algorithms/    # MAPPO, GAE, buffer dan normalisasi
src/training/      # Trainer, konfigurasi dan logging
src/evaluation/    # Evaluasi, checkpoint dan video
tests/             # Lima modul pengujian Easy/MAPPO
~~~

## Verifikasi

~~~bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
~~~

Tes berfokus pada reset tetap, kontak fisik, seluruh footprint, terminal/timeout,
Parallel API, reward, likelihood PPO, GAE, checkpoint/resume, TensorBoard dan video.
Tes tersebut memvalidasi implementasi; demo referensi dan smoke training tidak
merupakan bukti bahwa policy sudah belajar menyelesaikan task.
