clear
sin(acos(-1/3)/2)

s2c = @(rho,theta,phi) [rho*sin(theta)*cos(phi) rho*sin(theta)*sin(phi) rho*cos(theta)];
clf
figure(1)
XYZr = [s2c(1,0,0);
       s2c(tan(pi/6),acos(-1/3)/2,0);
       s2c(1/3,acos(-1/3)/2,pi/3);
       s2c(tan(pi/6),acos(-1/3)/2,2*pi/3);
       
       s2c(1,0,0);
       s2c(tan(pi/6),acos(-1/3)/2,2*pi/3);
       s2c(1/3,acos(-1/3)/2,pi);
       s2c(tan(pi/6),acos(-1/3)/2,4*pi/3);
       
       s2c(1,0,0);
       s2c(tan(pi/6),acos(-1/3)/2,4*pi/3);
       s2c(1/3,acos(-1/3)/2,-pi/3);
       s2c(tan(pi/6),acos(-1/3)/2,0);
       ];

Xr = XYZr(:,1);
Yr = XYZr(:,2);
Zr = XYZr(:,3);

XYZg = [s2c(1,acos(-1/3),0);
       s2c(tan(pi/6),acos(-1/3)/2,0);
       s2c(1/3,acos(-1/3)/2,pi/3);
       s2c(tan(pi/6),acos(tan(-pi/6)),pi/3);
       
       s2c(1,acos(-1/3),0);
       s2c(tan(pi/6),acos(tan(-pi/6)),pi/3);
       s2c(1/3,pi,0)
       s2c(tan(pi/6),acos(tan(-pi/6)),-pi/3);
       
       s2c(1,acos(-1/3),0);
       s2c(tan(pi/6),acos(tan(-pi/6)),-pi/3);
       s2c(1/3,acos(-1/3)/2,-pi/3);
       s2c(tan(pi/6),acos(-1/3)/2,0);
       ];

Xg = XYZg(:,1);
Yg = XYZg(:,2);
Zg = XYZg(:,3);

XYZb = [s2c(1,acos(-1/3),2*pi/3);
       s2c(tan(pi/6),acos(-1/3)/2,2*pi/3);
       s2c(1/3,acos(-1/3)/2,pi);
       s2c(tan(pi/6),acos(tan(-pi/6)),pi);
       
       s2c(1,acos(-1/3),2*pi/3);
       s2c(tan(pi/6),acos(tan(-pi/6)),pi);
       s2c(1/3,pi,0)
       s2c(tan(pi/6),acos(tan(-pi/6)),pi/3);
       
       s2c(1,acos(-1/3),2*pi/3);
       s2c(tan(pi/6),acos(tan(-pi/6)),pi/3);
       s2c(1/3,acos(-1/3)/2,pi/3);
       s2c(tan(pi/6),acos(-1/3)/2,2*pi/3);
       ];

Xb = XYZb(:,1);
Yb = XYZb(:,2);
Zb = XYZb(:,3);

XYZy = [s2c(1,acos(-1/3),4*pi/3);
       s2c(tan(pi/6),acos(-1/3)/2,4*pi/3);
       s2c(1/3,acos(-1/3)/2,-pi/3);
       s2c(tan(pi/6),acos(tan(-pi/6)),-pi/3);
       
       s2c(1,acos(-1/3),4*pi/3);
       s2c(tan(pi/6),acos(tan(-pi/6)),-pi/3);
       s2c(1/3,pi,0)
       s2c(tan(pi/6),acos(tan(-pi/6)),pi);
       
       s2c(1,acos(-1/3),4*pi/3);
       s2c(tan(pi/6),acos(tan(-pi/6)),pi);
       s2c(1/3,acos(-1/3)/2,pi);
       s2c(tan(pi/6),acos(-1/3)/2,4*pi/3);
       ];

Xy = XYZy(:,1);
Yy = XYZy(:,2);
Zy = XYZy(:,3);

XYZk = [s2c(0,0,0);
        s2c(1/3,acos(-1/3)/2,-pi/3);
        s2c(tan(pi/6),acos(tan(-pi/6)),-pi/3);
        s2c(1/3,pi,0);

        s2c(0,0,0);
        s2c(1/3,acos(-1/3)/2,pi/3);
        s2c(tan(pi/6),acos(tan(-pi/6)),pi/3);
        s2c(1/3,pi,0);

        s2c(0,0,0);
        s2c(1/3,acos(-1/3)/2,pi);
        s2c(tan(pi/6),acos(tan(-pi/6)),pi);
        s2c(1/3,pi,0);

        s2c(0,0,0);
        s2c(1/3,acos(-1/3)/2,-pi/3);
        s2c(tan(pi/6),acos(-1/3)/2,4*pi/3);
        s2c(1/3,acos(-1/3)/2,pi);

        s2c(0,0,0);
        s2c(1/3,acos(-1/3)/2,pi/3);
        s2c(tan(pi/6),acos(-1/3)/2,0);
        s2c(1/3,acos(-1/3)/2,-pi/3);

        s2c(0,0,0);
        s2c(1/3,acos(-1/3)/2,pi);
        s2c(tan(pi/6),acos(-1/3)/2,2*pi/3);
        s2c(1/3,acos(-1/3)/2,pi/3);
       ];

Xk = XYZk(:,1);
Yk = XYZk(:,2);
Zk = XYZk(:,3);

hold on
patch(Xr,Yr,Zr,'r','FaceAlpha',.5)
for k=1:3
    patch(Xg(4*k-3:4*k),Yg(4*k-3:4*k),Zg(4*k-3:4*k),'g','FaceAlpha',.5)
    patch(Xb(4*k-3:4*k),Yb(4*k-3:4*k),Zb(4*k-3:4*k),'b','FaceAlpha',.5)
    patch(Xy(4*k-3:4*k),Yy(4*k-3:4*k),Zy(4*k-3:4*k),'y','FaceAlpha',.5)
end
for k=1:6
    patch(Xk(4*k-3:4*k),Yk(4*k-3:4*k),Zk(4*k-3:4*k),'k','FaceAlpha',.5)
end
hold off


axis equal
grid off
view(120,15)
set(gca,'XColor','none','Ycolor','none','Zcolor','none')
saveas(gcf,'tetrahedron.png')