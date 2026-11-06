# T-Shape MAPPO — Easy dan Medium

Baseline kecil untuk belajar cooperative transport dengan MuJoCo dan shared-policy MAPPO.
Proyek saat ini menjalankan **arena Easy, dengan 1–16 agen dan posisi awal agen
yang diacak dekat objek**. Default empat agen. Posisi/yaw objek dan target tetap
sama pada setiap episode.

Arena Easy tidak memiliki penghalang; Medium memakai satu pintu pada sekat arena.
Tidak ada curriculum otomatis. Setiap jumlah agen dilatih sebagai
policy tersendiri. Checkpoint baseline Fixed Easy sebelumnya tetap dapat dievaluasi
dengan susunan empat agen tetap yang tersimpan di dalamnya.

## Skenario Easy (default)

| Komponen | Konfigurasi awal |
| --- | --- |
| Ruang arena | 5,8 x 5,8 m, tanpa rintangan internal |
| Objek | Dua palang asimetris dan satu batang, satu rigid body |
| Dimensi objek | Panjang 1,2 m; palang besar 0,6 m; palang kecil 0,3 m |
| Massa objek | 2 kg |
| Posisi / yaw objek | (-1,5, -1,5), 0 rad — kiri bawah |
| Agen | Default empat; 1–16 silinder, radius 0,03 m, massa 0,25 kg per agen |
| Posisi agen | Acak di sekitar seluruh permukaan objek, gap 4–12 cm |
| Pusat target | (1,5, 1,5) — kanan atas |
| Ukuran penuh target | 1,5 x 1,3 m |
| Horizon | 500 keputusan, setara 25 detik simulasi |
| Kontrol | Dua gaya kontinu per agen, maksimal +/-3 N per sumbu |

Objek mulai di kiri bawah dan target tetap di kanan atas. Agen ditempatkan pada
pita di luar footprint objek, termasuk palang pendek, dengan gap permukaan 4–12 cm.
Sampling menolak posisi di dalam objek, di luar arena, atau terlalu dekat agen lain
(jarak antarpermukaan minimal 1 cm). Sisi awal dan identitas agen tidak diberi peran
dorong tertentu. Gerak X/Y dan rotasi objek mengikuti fisika.

Gerak planar memakai joint X/Y dan yaw objek dengan gravitasi nol. Lantai bersifat
visual. Resistansi objek berasal dari damping dan frictionloss joint, bukan beban
normal lantai. Parameter fisika berada di [arena.xml](assets/mujoco/arena.xml).

Setiap keputusan menahan aksi selama 50 substep x 0,001 s = 0,05 s, sehingga policy
berjalan pada 20 Hz. Setiap reset mengosongkan kecepatan/gaya dan mengambil susunan
agen baru. Seed mengendalikan jaringan, sampling aksi, PPO, dan urutan susunan agen.
Seed yang sama mereproduksi urutan episode; trainer tidak mengulang seed tiap reset.

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
Demo selalu memakai susunan empat agen tetap lama; bukan uji policy pada spawn acak.

## Medium: satu pintu

Gunakan [medium.yaml](configs/medium.yaml) dan [arena_medium.xml](assets/mujoco/arena_medium.xml).
Sekat berada pada X = 0, tebal 0,15 m, dengan satu bukaan selebar 0,90 m di Y = 0.
Kedua ujung sekat menyambung ke dinding arena sehingga tidak ada jalur memutar.
Objek mulai pada (-1,5, 0), yaw 0; empat agen default diacak di dekat objek.
Objek memanjang 1,2 m dan memiliki palang maksimum 0,6 m: ia perlu berputar
untuk lewat. Pada yaw pi/2, palang memiliki clearance 0,15 m per sisi bukaan.
Geometri, bukan controller tersembunyi, menentukan apakah objek muat.

Sukses terjadi ketika **seluruh sudut geometri objek** sudah berada di kanan
permukaan keluar sekat ditambah margin 0,05 m (minimum X objek >= 0,125 m).
Pose akhir, kecepatan rendah, dan containment pada kotak target tidak disyaratkan.
Area hijau menunjukkan sisi tujuan. Sukses langsung mengakhiri episode sebagai
termination, termasuk pada langkah terakhir; timeout tanpa sukses tetap truncation.
Kegagalan fisika diprioritaskan jika terjadi pada langkah yang sama.

Reward Medium adalah reward tim bersama:

~~~text
distance = 0.1 * (1 - tanh(d / 2.0))
progress = 2 * (previous_d - d)
success  = 20 pada keberhasilan menyeberang
reward   = distance + progress + success
~~~

`d` adalah panjang lintasan titik pusat massa menuju referensi (1,5, 0), melalui
bukaan. Perhitungan memakai lintasan lurus atau sudut bukaan tanpa pergantian fase
diskontinu. Ini panduan reward untuk pusat objek, bukan jaminan clearance seluruh
footprint atau perintah gerak. `distance_to_goal` di log Medium memakai jarak ini.
Reward jarak positif, meningkat saat mendekat, dan dibatasi bobotnya; komponen
progress tetap dapat negatif ketika menjauh. Tidak ada reward orientasi.
Kegagalan fisika mengganti semua komponen normal dengan -20. Bonus sukses dan
episode yang selesai saat menyeberang membatasi kesempatan mengumpulkan reward
dengan diam dekat tujuan, tetapi keberhasilan belajar tetap perlu dievaluasi.

Timestep 0,001 s, frame skip 50, gaya maksimum 3 N per sumbu per agen, arsitektur
jaringan, dan hyperparameter MAPPO tetap sama dengan Easy. Actor dan critic
Medium mendapat lima fitur tambahan: posisi pusat pintu (relatif untuk actor),
ketebalan sekat, lebar bukaan, dan posisi bidang sukses sepanjang X (relatif untuk
actor). Dimensinya 23+4N dan 22+4N, atau 39 dan 38 untuk empat agen.
Checkpoint Medium menggunakan task `medium_v1`; evaluasi/resume memulihkan task
dan XML tersimpan. Checkpoint Easy tetap didukung, tetapi tidak dapat langsung
dipakai sebagai policy Medium karena dimensi input berbeda.

~~~bash
python scripts/train.py --env-config configs/medium.yaml --device cuda --num-agents 4 --total-steps 1000000 --run-dir results/medium_n4_s42
python scripts/evaluate.py results/medium_n4_s42/checkpoints/best.pt --device auto --episodes 20 --seed 2000000
python scripts/evaluate.py results/medium_n4_s42/checkpoints/best.pt --device auto --render
python scripts/train.py --resume results/medium_n4_s42/checkpoints/latest.pt --device cuda --total-steps 2000000 --run-dir results/medium_n4_s42_resume
~~~

Gunakan run directory baru. Log episode/evaluasi mencatat `crossing_clearance_m`,
geometri pintu pada `initial_state`, dan `episode_metrics.obstacle_collision_events`
secara terpisah dari kontak dinding luar. `completion_time_s` adalah waktu melintas.
`scripts/demo.py` tetap controller referensi untuk Easy.

## Training dari awal

~~~bash
python scripts/train.py --device cuda --num-agents 4 --total-steps 1000000 --run-dir results/easy_random_n4
~~~

Gunakan device cpu untuk komputer tanpa CUDA. Run directory harus baru.
Bobot actor dan critic diinisialisasi dari awal.

Tanpa video otomatis:

~~~bash
python scripts/train.py --device cpu --num-agents 4 --total-steps 100000 --video-interval-steps 0 --run-dir results/easy_random_n4_cpu
~~~

Perbandingan jumlah agen dengan budget dan seed yang sama, dijalankan berurutan:

~~~bash
for n in 2 4 8 16; do
  python scripts/train.py --device cuda --num-agents "$n" --seed 42 --total-steps 1000000 --run-dir "results/easy_random_n${n}_s42"
done
~~~

Untuk membandingkan randomisasi dengan susunan awal tetap:

~~~bash
python scripts/train.py --device cuda --num-agents 4 --no-randomize-agents --seed 42 --total-steps 1000000 --run-dir results/easy_fixed_n4_s42
~~~

Tanpa randomisasi, empat agen menggunakan susunan awal XML lama. Untuk jumlah
lain, layout dibuat sekali secara deterministik dengan seed 0 dan diulang tiap
episode. `agent_positions` dapat menentukan koordinat tetap secara eksplisit,
sebanyak `num_agents` pasangan XY; koordinat ini hanya digunakan ketika
`randomize_agents: false`.

Konfigurasi berada di [environment.yaml](configs/environment.yaml), termasuk
`num_agents`, `randomize_agents`, `spawn_gap_min` dan `spawn_gap_max`; hyperparameter
ada di [mappo.yaml](configs/mappo.yaml). Argumen CLI menimpa nilai YAML.

Gaya maksimum tetap 3 N per sumbu **per agen**. Menambah jumlah agen juga menambah
total kapasitas gaya tim. Laporkan faktor ini saat membandingkan performa;
eksperimen ini belum memisahkan efek jumlah agen dari kapasitas gaya total.

Trainer menggunakan satu arena. MuJoCo berjalan pada CPU; jaringan dapat berjalan
pada CPU atau GPU.

## Evaluasi

~~~bash
python scripts/evaluate.py results/easy_random_n4/checkpoints/best.pt --device auto --episodes 20 --seed 2000000
python scripts/evaluate.py results/easy_random_n4/checkpoints/best.pt --device auto --render
python scripts/evaluate.py results/easy_random_n4/checkpoints/latest.pt --device auto --video-dir results/easy_random_n4/videos/manual --json results/easy_random_n4/manual_evaluation.json
~~~

Rekaman pertama disimpan sebagai policy.mp4 di video directory. Hanya episode
pertama direkam, termasuk ketika jumlah episode evaluasi lebih dari satu.

CLI evaluasi default memakai satu episode dengan aksi deterministik. Jumlah agen
dan aturan spawn mengikuti checkpoint; tidak perlu argumen jumlah agen saat evaluasi.
Pada checkpoint dengan spawn acak, episode berbeda memakai seed berbeda sehingga
posisi awal berbeda. Mengulang command dengan seed yang sama mereproduksi susunan.
Untuk melihat variasi tambahan akibat sampling aksi:

~~~bash
python scripts/evaluate.py results/easy_random_n4/checkpoints/best.pt --stochastic --episodes 10
~~~

Script mencetak progres saat memuat policy, memulai episode, dan selama evaluasi.

## Resume training baru ini

~~~bash
python scripts/train.py --resume results/easy_random_n4/checkpoints/latest.pt --device cuda --total-steps 2000000 --run-dir results/easy_random_n4_resume
~~~

Budget bersifat kumulatif: contoh tersebut melanjutkan sampai total 2.000.000 timestep.
Resume memulihkan model, optimizer, normalisasi dan RNG, termasuk RNG spawn agen.
Simulasi memulai episode baru; state fisika episode yang terputus tidak dipulihkan.
Output ditulis ke directory baru, dengan best yang dipilih dari evaluasi run lanjutan.
Jumlah agen dan aturan spawn dipulihkan dari checkpoint dan tidak dapat diubah
saat resume. Untuk jumlah agen baru, mulai training baru.

Checkpoint Easy memakai format 2. Checkpoint Fixed Easy sebelumnya tetap didukung;
checkpoint format eksperimen yang lebih lama tidak dimigrasikan.

## Observasi, state dan aksi

Actor menerima **18 + 4N fitur per agen** (34 saat N=4): posisi/kecepatan diri, posisi objek dan target
relatif terhadap diri, sin/cos yaw objek, kecepatan objek, ukuran target, posisi
relatif serta kecepatan N−1 agen lain, jarak ke batas luar arena, kontak dengan
objek, waktu tersisa dan progres hold sukses.

Observasi masih menggunakan informasi global. Tidak ada batas jangkauan sensor.
Posisi relatif memakai sumbu dunia; kecepatan agen lain masih absolut.

Critic menerima **17 + 4N fitur global** (33 saat N=4): pose/kecepatan objek, target,
seluruh posisi/kecepatan agen, batas arena, waktu tersisa dan progres hold.
Actor bersama: (18+4N) -> 128 -> 128 -> 2; critic: (17+4N) -> 256 -> 256 -> 1,
dengan aktivasi Tanh. Ukuran input berbeda antarjumlah agen, sehingga checkpoint
tidak digunakan untuk transfer lintas jumlah agen.

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
success     = 1 hanya saat syarat sukses pertama kali terpenuhi dalam episode

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

Sukses tidak mengakhiri episode. Bonus +20 diberikan sekali pada pencapaian pertama;
episode terus berjalan sampai batas langkah. `success` mencatat bahwa episode pernah
berhasil, dan `completion_time_s` menyimpan waktu pencapaian pertama, meskipun objek
kemudian keluar target. Fitur observasi/state `goal_hold_fraction` tetap 1 setelah
pencapaian pertama agar policy mengetahui bonus sudah diterima. Sebelum itu, fitur
ini menunjukkan kemajuan hold saat ini. `goal_hold_steps` pada info tetap menghitung
hold aktual dan kembali nol ketika objek keluar atau belum tenang.

Goal yaw merupakan preferensi reward; tidak ada toleransi sudut tambahan pada
syarat sukses. Timeout adalah truncation, termasuk episode yang pernah berhasil.
Hanya kegagalan fisika yang menjadi termination; episode tersebut dilaporkan gagal
meskipun sebelumnya sempat mencapai target. GAE melakukan bootstrap pada timeout
dari state akhir sebelum reset. Evaluasi sekarang memakai aturan ini juga untuk
checkpoint lama, sehingga return dan panjang episode dapat berbeda dari log lama.

Pemeriksaan setiap physics substep mendeteksi state non-finite, peringatan MuJoCo,
penetrasi >5 mm, linear speed >5 m/s, dan yaw speed >10 rad/s.

## Logging dan file hasil

Default: rollout 1.024 timestep tim, PPO lima epoch, gamma 0,99, GAE lambda 0,95,
learning rate actor/critic 0,0003. Satu timestep menerapkan aksi seluruh N agen bersama.

Evaluasi berlangsung setiap 10.000 timestep dan saat budget berakhir. Untuk spawn
acak, policy menggunakan aksi deterministik pada sepuluh layout dengan seed
1.000.000–1.000.009 yang sama di setiap evaluasi. Ini validasi untuk memilih best;
gunakan seed lain untuk evaluasi akhir. `--eval-episodes` mengatur jumlah layout
monitoring (default 10); evaluasi ini menambah waktu simulasi. Pada spawn tetap,
hanya satu episode diperlukan. Video merekam satu episode dengan seed 1.000.000,
sehingga hasilnya tidak selalu sama dengan success rate rata-rata sepuluh layout.
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
assets/mujoco/arena_medium.xml
configs/environment.yaml
configs/medium.yaml
configs/mappo.yaml
scripts/demo.py
scripts/train.py
scripts/evaluate.py
src/envs/          # Fisika, pintu Medium, model, spawn dan Parallel API
src/agents/        # Shared actor dan centralized critic
src/algorithms/    # MAPPO, GAE, buffer dan normalisasi
src/training/      # Trainer, konfigurasi dan logging
src/evaluation/    # Evaluasi, checkpoint dan video
tests/             # Pengujian Easy/Medium dan MAPPO
~~~

## Verifikasi

~~~bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
~~~

Tes berfokus pada reset tetap/acak, populasi 1/2/4/8/16, kontak fisik, seluruh footprint, terminal/timeout,
Parallel API, reward, likelihood PPO, GAE, checkpoint/resume, TensorBoard dan video.
Tes tersebut memvalidasi implementasi; demo referensi dan smoke training tidak
merupakan bukti bahwa policy sudah belajar menyelesaikan task.
